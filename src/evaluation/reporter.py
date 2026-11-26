"""Local JSON provenance and checked internal plans; not the client's export format."""

import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

from configs import PROJECT_ROOT
from src.data.dataset import SOLUTION_COLUMNS, Mine
from src.evaluation.metrics import Plan


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


def create_run_directory(root: Path, label: str, name: str | None = None, now: datetime | None = None) -> Path:
    """root/<YYYY-MM-DD_HH-MM>_<label>[_<name>] in local time; if it already exists,
    _2, _3, ... is appended (never a random hash)."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    base = f"{stamp}_{label}" + (f"_{name}" if name else "")
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10_000):
        path = root / (base if index == 1 else f"{base}_{index}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise OSError(f"Demasiadas carpetas con el nombre {base} en {root}")


def write_rows_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """Rows -> CSV with the given columns; None is written as an empty cell."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(["" if row.get(c) is None else row.get(c) for c in columns] for row in rows)


def preparation_metadata(data_path: str) -> dict | None:
    """Provenance written by tools/prepare_blocks.py next to the CSV (<name>.meta.json), if any."""
    meta = Path(data_path).with_suffix(".meta.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None


def mine_metadata(mine: Mine) -> dict:
    return {"path": mine.source, "sha256": mine.data_sha256,
            "preparation": preparation_metadata(mine.source) if mine.source != "synthetic" else None,
            "block_count": len(mine.blocks), "reachable_blocks": sum(b.reachable for b in mine.blocks)}


def write_plan_csv(path: Path, plan: Plan) -> None:
    """Ordered plan -> standard plan CSV (order,id,period), the same format as a solution."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_COLUMNS)
        writer.writerows((order, e.block_id, e.period) for order, e in enumerate(plan.extractions, start=1))

