"""Local JSON provenance and checked internal plans; not the client's export format."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

from src.config import PROJECT_ROOT
from src.data.dataset import Mine
from src.data.transforms import Precedences
from src.evaluation.metrics import Extraction, Plan, evaluate_plan


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=all")
    files = [PROJECT_ROOT / name for name in ("main.py", "environment.yml", "README.md")]
    for directory in ("src", "tests", "docs", ".github"):
        files.extend(p for p in (PROJECT_ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".md", ".yml"})
    hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p.read_bytes()).hexdigest()
              for p in sorted(files) if p.is_file()}
    snapshot = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "status": status, "source_sha256": snapshot, "file_hashes": hashes}


def runtime_metadata() -> dict:
    return {"python": sys.version, "platform": platform.platform(), "openpyxl": version("openpyxl")}


def create_run_directory(root: Path, mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = root / f"{stamp}-{mode}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def mine_metadata(mine: Mine) -> dict:
    return {"path": mine.source, "sha256": mine.data_sha256, "sheet": mine.sheet,
            "block_count": len(mine.blocks), "economic_units": "No declaradas en el libro; se conserva Valor",
            "blocks": [{"internal_index": i, **asdict(b)} for i, b in enumerate(mine.blocks)]}


def precedence_metadata(precedences: Precedences | None) -> dict | None:
    if precedences is None:
        return None
    return {"rule": precedences.rule, "source": precedences.source, "provisional": precedences.provisional,
            "predecessors": {str(k): sorted(v) for k, v in precedences.predecessors.items()},
            "within_period": "Predecessors must occur earlier in the ordered plan; same period allowed"}


def plan_payload(plan: Plan) -> dict:
    return {"schema_version": 1, "format": "openpit-internal-ordered-plan",
            "period_origin": 0, "extractions": [asdict(e) for e in plan.extractions]}


def load_plan(path: Path) -> Plan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("El plan JSON debe ser un objeto")
    if payload.get("schema_version") != 1 or payload.get("format") != "openpit-internal-ordered-plan":
        raise ValueError("Formato de plan interno desconocido")
    if payload.get("period_origin") != 0:
        raise ValueError("El formato interno requiere períodos desde cero")
    entries = payload.get("extractions")
    if not isinstance(entries, list):
        raise ValueError("extractions debe ser una lista ordenada")
    if any(not isinstance(e, dict) or set(e) != {"block_id", "period"} for e in entries):
        raise ValueError("Cada extracción requiere exactamente block_id y period")
    return Plan(tuple(Extraction(e["block_id"], e["period"]) for e in entries))


def export_checked_plan(directory: Path, mine: Mine, plan: Plan, parameters,
                        precedences: Precedences | None) -> str | None:
    evaluation = evaluate_plan(mine, plan, parameters, precedences)
    if evaluation.feasible is not True:
        return None
    name = "conditional-plan.json" if evaluation.precedence_status == "conditional" else "verified-plan.json"
    write_json(directory / name, {**plan_payload(plan), "evaluation": asdict(evaluation),
                                 "data_sha256": mine.data_sha256,
                                 "domain_validation": "pending-client-confirmation"})
    return name


def q_table_payload(model) -> dict:
    def key_payload(key):
        payload = asdict(key)
        if "extracted" in payload:
            payload["extracted"] = sorted(payload["extracted"])
        return payload
    return {"model": model.name, "entries": [
        {"state": key_payload(state), "action": action, "q": value}
        for (state, action), value in model.q.items()
    ]}
