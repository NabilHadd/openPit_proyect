"""Extract Aldo's reference solution from his Excel workbook into the standard solution CSV.

Auxiliary tool, outside main.py, deliberately specific to that workbook:
- block table in sheet "105 Bloques" (ID, X, Y, Z);
- solution painted in sheet "Resultado": cells whose fill equals the legend cell K3,
  laid out in panels labelled "Z=<level>" (7 columns of X, 5 rows of Y).

    python -m tools.extract_aldo_solution data/raw/data.xlsx

Writes <out> (order,id,period; order empty = unordered set, period 0) and <out>.meta.json.
"""

import argparse
import colorsys
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python tools/extract_aldo_solution.py` as well as `python -m`
    sys.path.insert(0, str(ROOT))

from src.config import project_path  # noqa: E402
from src.data.dataset import SOLUTION_COLUMNS, Block, Mine, file_hash, integer  # noqa: E402

BLOCK_SHEET = "105 Bloques"
SOLUTION_SHEET = "Resultado"
LEGEND_CELL = "K3"
DEFAULT_OUTPUT = "data/processed/solution.csv"


def color_rgb(color, workbook) -> str:
    if color.type == "rgb":
        rgb = color.rgb[-6:]
    elif color.type == "theme":
        if not workbook.loaded_theme:
            raise ValueError("Color de tema sin tema Excel")
        root = ET.fromstring(workbook.loaded_theme)
        scheme = root.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}clrScheme")
        if scheme is None or color.theme >= len(scheme):
            raise ValueError("Color de tema desconocido")
        element = list(scheme[color.theme])[0]
        rgb = element.attrib.get("lastClr", element.attrib.get("val"))
    elif color.type == "indexed" and color.indexed < len(workbook._colors):
        rgb = workbook._colors[color.indexed][-6:]
    else:
        raise ValueError(f"Color no resoluble: {color}")
    channels = [int(rgb[i:i + 2], 16) for i in (0, 2, 4)]
    tint = color.tint
    if tint:
        # SpreadsheetML tint changes HLS luminance, not each RGB channel.
        hue, luminance, saturation = colorsys.rgb_to_hls(*(c / 255 for c in channels))
        luminance = luminance * (1 + tint) if tint < 0 else luminance * (1 - tint) + tint
        channels = [round(c * 255) for c in colorsys.hls_to_rgb(hue, luminance, saturation)]
    return "".join(f"{c:02X}" for c in channels)


def fill_signature(cell, workbook) -> tuple:
    fill = cell.fill
    if fill.patternType != "solid":
        raise ValueError(f"{cell.coordinate}: solo se admiten rellenos sólidos")
    return fill.patternType, color_rgb(fill.fgColor, workbook)


@dataclass(frozen=True)
class Selection:
    sheet: str
    ids: tuple[int, ...]
    cells: tuple[str, ...]
    legend_cell: str = LEGEND_CELL


def read_block_table(workbook) -> Mine:
    if BLOCK_SHEET not in workbook.sheetnames:
        raise ValueError(f"Falta la hoja {BLOCK_SHEET!r}")
    rows = workbook[BLOCK_SHEET].iter_rows(values_only=True)
    headers = next(rows, ())
    required = ("ID", "X", "Y", "Z")
    if any(headers.count(name) != 1 for name in required):
        raise ValueError(f"Encabezados obligatorios y únicos en {BLOCK_SHEET!r}: {required}")
    index = {name: headers.index(name) for name in required}
    blocks = tuple(
        Block(*(integer(row[index[name]], f"{name} fila {n}") for name in required), value=0.0, source_row=n)
        for n, row in enumerate(rows, start=2) if not all(v is None for v in row))
    return Mine(blocks)


def read_selection(workbook, sheet_name: str, mine: Mine) -> Selection:
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Falta la hoja visual {sheet_name}")
    sheet = workbook[sheet_name]
    legend = fill_signature(sheet[LEGEND_CELL], workbook)
    by_position = {(b.x, b.y, b.z): b.id for b in mine.blocks}
    ids, cells, seen = [], [], set()
    # This adapter recognizes the workbook's labelled level panels, not style IDs.
    panels = []
    for row in sheet:
        for cell in row:
            if cell.column == 1 and isinstance(cell.value, str) and cell.value.startswith("Z="):
                try:
                    z = int(cell.value[2:])
                except ValueError as exc:
                    raise ValueError(f"Nivel no reconocido: {cell.value}") from exc
                header_row = cell.row + 1 if cell.row == 1 else cell.row
                panels.append((z, header_row))
    if not panels:
        raise ValueError(f"{sheet_name}: no se encontraron paneles Z=")
    for z, header_row in panels:
        for row in range(header_row + 1, header_row + 6):
            y = integer(sheet.cell(row, 2).value, f"{sheet_name}:Y fila {row}")
            for column in range(3, 10):
                x = integer(sheet.cell(header_row, column).value, f"{sheet_name}:X columna {column}")
                cell = sheet.cell(row, column)
                block_id = integer(cell.value, f"{sheet_name}!{cell.coordinate}")
                if by_position.get((x, y, z)) != block_id:
                    raise ValueError(f"{sheet_name}!{cell.coordinate}: ID/coordenadas no corresponden a tabla")
                if block_id in seen:
                    raise ValueError(f"Bloque repetido en hoja visual: {block_id}")
                seen.add(block_id)
                if cell.fill.patternType and fill_signature(cell, workbook) == legend:
                    ids.append(block_id)
                    cells.append(cell.coordinate)
    if seen != set(mine.by_id):
        raise ValueError(f"{sheet_name}: la vista no corresponde al yacimiento completo")
    return Selection(sheet_name, tuple(ids), tuple(cells))


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def extract(source: Path, output: Path) -> tuple[Path, Path]:
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    workbook = load_workbook(source, data_only=True)
    try:
        selection = read_selection(workbook, SOLUTION_SHEET, read_block_table(workbook))
    finally:
        workbook.close()
    if not selection.ids:
        raise ValueError(f"{SOLUTION_SHEET}: no hay celdas con el color de la leyenda {LEGEND_CELL}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_COLUMNS)
        writer.writerows(("", block_id, 0) for block_id in sorted(selection.ids))
    meta_path = output.with_suffix(".meta.json")
    meta = {
        "schema": "openpit-solution", "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/extract_aldo_solution.py",
        "source": {"path": display_path(source), "sha256": file_hash(source), "sheet": SOLUTION_SHEET,
                   "legend_cell": LEGEND_CELL},
        "ordered": False, "period_note": "El Excel no indica períodos; se asigna period=0",
        "output": {"path": display_path(output), "sha256": file_hash(output), "rows": len(selection.ids)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, meta_path


def main(argv=None):
    cli = argparse.ArgumentParser(description="Excel de Aldo (hoja Resultado) -> CSV estándar de solución")
    cli.add_argument("source", help="Excel de Aldo; relativo al repositorio o absoluto")
    cli.add_argument("--out", default=DEFAULT_OUTPUT, help=f"CSV de salida; default={DEFAULT_OUTPUT}")
    args = cli.parse_args(argv)
    try:
        output, meta = extract(project_path(args.source), project_path(args.out))
    except (ValueError, OSError, KeyError) as exc:
        cli.exit(2, f"Error: {exc}\n")
    print(f"Solución: {output}\nMetadatos: {meta}")


if __name__ == "__main__":
    main()
