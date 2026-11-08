"""Local JSON provenance and checked internal plans; not the client's export format."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

from src.config import PROJECT_ROOT
from src.data.dataset import SOLUTION_COLUMNS, Mine
from src.data.precedences import Precedences
from src.evaluation.metrics import Plan, evaluate_plan


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=all")
    files = [PROJECT_ROOT / name for name in ("main.py", "environment.yml")]
    for directory in ("src", "configs"):
        files.extend(p for p in (PROJECT_ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".md", ".yml"})
    hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p.read_bytes()).hexdigest()
              for p in sorted(files) if p.is_file()}
    snapshot = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "source_sha256": snapshot}


def runtime_metadata() -> dict:
    return {"python": sys.version, "platform": platform.platform()}


def create_run_directory(root: Path, mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = root / f"{stamp}-{mode}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def preparation_metadata(data_path: str) -> dict | None:
    """Provenance written by tools/prepare_blocks.py next to the CSV (<name>.meta.json), if any."""
    meta = Path(data_path).with_suffix(".meta.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None


def mine_metadata(mine: Mine) -> dict:
    return {"path": mine.source, "sha256": mine.data_sha256,
            "preparation": preparation_metadata(mine.source) if mine.source != "synthetic" else None,
            "block_count": len(mine.blocks), "reachable_blocks": sum(b.reachable for b in mine.blocks)}


PLAN_FILE = "plan.csv"


def write_plan_csv(path: Path, plan: Plan) -> None:
    """Ordered plan -> standard plan CSV (order,id,period), the same format as a solution."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_COLUMNS)
        writer.writerows((order, e.block_id, e.period) for order, e in enumerate(plan.extractions, start=1))


def export_checked_plan(directory: Path, mine: Mine, plan: Plan, parameters,
                        precedences: Precedences | None) -> str | None:
    """Write the plan only if the independent verifier accepts it. Its evaluation belongs
    in the run's report.json, not in the CSV."""
    evaluation = evaluate_plan(mine, plan, parameters, precedences)
    if evaluation.feasible is not True:
        return None
    write_plan_csv(directory / PLAN_FILE, plan)
    return PLAN_FILE


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
