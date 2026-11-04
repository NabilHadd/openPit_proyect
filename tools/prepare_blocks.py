"""Convert a raw block table (CSV or Excel) into the pipeline's standard blocks CSV.

Auxiliary tool, outside main.py. Nothing about the source format is hard-coded:
the sheet, the column names and the orientation of z are arguments.

    python -m tools.prepare_blocks data/raw/data.xlsx --surface z-max --sheet "105 Bloques" \\
        --map ID=id X=x Y=y Z=z Valor=value

The pipeline assumes higher z = shallower. With `--surface z-min` (the source numbers
levels from the surface downwards) z is inverted: z' = z_min + z_max - z.

Writes <out> (standard CSV: id,x,y,z,value[,tonnes]) and <out>.meta.json (provenance).
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python tools/prepare_blocks.py` as well as `python -m`
    sys.path.insert(0, str(ROOT))

from src.config import project_path  # noqa: E402
from src.data.dataset import BLOCK_COLUMNS, OPTIONAL_BLOCK_COLUMNS, file_hash, load_mine  # noqa: E402

STANDARD_COLUMNS = BLOCK_COLUMNS + OPTIONAL_BLOCK_COLUMNS
INTEGER_COLUMNS = ("id", "x", "y", "z")
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
DEFAULT_OUTPUT = "data/processed/data.csv"
SURFACES = ("z-max", "z-min")


def parse_column_map(items: list[str]) -> dict[str, str]:
    """["ID=id", "Valor=value"] -> {"ID": "id", "Valor": "value"}."""
    mapping = {}
    for item in items:
        source, separator, target = item.partition("=")
        if not separator or not source or target not in STANDARD_COLUMNS:
            raise ValueError(f"Mapeo inválido {item!r}: use ORIGEN=destino con destino en {STANDARD_COLUMNS}")
        mapping[source] = target
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Dos columnas de origen apuntan a la misma columna estándar")
    return mapping


def read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if sheet is not None:
            raise ValueError("--sheet solo aplica a archivos Excel")
        return pd.read_csv(path)
    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    raise ValueError(f"Formato no soportado {suffix!r}; use .csv o Excel ({', '.join(sorted(EXCEL_SUFFIXES))})")


def standardize(frame: pd.DataFrame, column_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Rename to standard columns, validate every row and return (clean table, dropped columns)."""
    missing_sources = [c for c in column_map if c not in frame.columns]
    if missing_sources:
        raise ValueError(f"Columnas de origen inexistentes: {missing_sources}; disponibles: {list(frame.columns)}")
    frame = frame.rename(columns=column_map)
    if frame.columns.duplicated().any():
        raise ValueError(f"Columnas repetidas tras el mapeo: {list(frame.columns)}")
    missing = [c for c in BLOCK_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias {missing}; indíquelas con --map ORIGEN=destino")
    keep = [c for c in STANDARD_COLUMNS if c in frame.columns]
    dropped = [str(c) for c in frame.columns if c not in keep]
    frame = frame[keep].dropna(how="all")

    errors = []
    clean = {}
    rows = frame.index + 2  # row number in the source file (header = row 1)
    for column in keep:
        values = frame[column]
        is_bool = values.map(lambda v: isinstance(v, bool))
        numeric = pd.to_numeric(values.where(~is_bool), errors="coerce")
        finite = numeric.map(lambda v: not pd.isna(v) and math.isfinite(v))
        invalid = ~finite & (values.notna() if column == "tonnes" else True)
        if column in INTEGER_COLUMNS:
            invalid |= finite & (numeric % 1 != 0)
        if column == "tonnes":
            invalid |= finite & (numeric <= 0)
        if invalid.any():
            examples = [f"fila {r}: {v!r}" for r, v in zip(rows[invalid.to_numpy()], values[invalid])][:5]
            errors.append(f"{column}: valores inválidos ({int(invalid.sum())}) -> {examples}")
        clean[column] = numeric
    if errors:
        raise ValueError("Datos inválidos:\n  " + "\n  ".join(errors))

    table = pd.DataFrame(clean)
    for column in INTEGER_COLUMNS:
        table[column] = table[column].astype("int64")
    for key in (["id"], ["x", "y", "z"]):
        duplicated = table.duplicated(key, keep=False)
        if duplicated.any():
            raise ValueError(f"{'/'.join(key)} repetidos en filas {list(rows[duplicated.to_numpy()])[:10]}")
    if table.empty:
        raise ValueError("La tabla no contiene bloques")
    return table, dropped


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def orient(table: pd.DataFrame, surface: str) -> pd.DataFrame:
    """Return the table with the pipeline's convention: higher z = shallower."""
    if surface not in SURFACES:
        raise ValueError(f"surface debe ser una de {SURFACES}")
    if surface == "z-min":
        table = table.assign(z=table["z"].min() + table["z"].max() - table["z"])
    return table


def prepare(source: Path, output: Path, column_map: dict[str, str], surface: str,
            sheet: str | None = None) -> tuple[Path, Path]:
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    table, dropped = standardize(read_table(source, sheet), column_map)
    table = orient(table, surface)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    mine = load_mine(output)  # the produced file must be readable by the pipeline
    meta_path = output.with_suffix(".meta.json")
    meta = {
        "schema": "openpit-blocks", "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/prepare_blocks.py",
        "source": {"path": display_path(source), "sha256": file_hash(source), "sheet": sheet},
        "column_map": column_map, "dropped_columns": dropped,
        "surface_in_source": surface, "z_inverted": surface == "z-min",
        "output": {"path": display_path(output), "sha256": mine.data_sha256, "rows": len(mine.blocks)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, meta_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Tabla de bloques (CSV/Excel) -> CSV estándar del pipeline")
    result.add_argument("source", help="Archivo de origen (.csv, .xlsx, .xlsm, .xls); relativo al repositorio o absoluto")
    result.add_argument("--surface", required=True, choices=SURFACES,
                        help="Dónde está la superficie en el archivo de origen: z-max (Z mayor) o z-min (Z menor)")
    result.add_argument("--sheet", help="Hoja del Excel; por defecto la primera")
    result.add_argument("--map", nargs="+", default=[], metavar="ORIGEN=destino",
                        help=f"Renombrar columnas al estándar {STANDARD_COLUMNS}")
    result.add_argument("--out", default=DEFAULT_OUTPUT, help=f"CSV de salida; default={DEFAULT_OUTPUT}")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        output, meta = prepare(project_path(args.source), project_path(args.out), parse_column_map(args.map),
                               args.surface, args.sheet)
    except (ValueError, OSError, KeyError) as exc:
        cli.exit(2, f"Error: {exc}\n")
    print(f"Bloques: {output}\nMetadatos: {meta}")


if __name__ == "__main__":
    main()
