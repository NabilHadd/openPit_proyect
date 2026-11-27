"""Extract a reference solution painted in an Excel workbook into the standard solution CSV.

Auxiliary tool, outside main.py. It reads the workbook layout the client uses, without
fixing its size or its number of periods:
- block table: the sheet whose header has ID, X, Y, Z (or the one given with --blocks-sheet);
- solution sheet (default "Resultado"): one panel per level, labelled "Z=<level>" in
  column A, with the X values in a header row (from column C) and the Y values in column B;
  each cell holds a block ID and its fill colour says whether (and when) it is extracted;
- legend: every filled empty cell whose right neighbour is a text. If the texts name a
  period ("Bloques extraídos Período 2"), each colour is that period; otherwise there must
  be a single legend and every painted block goes to period 0.

Excel periods are numbered from 1; the CSV numbers them from 0 (Período 1 -> period 0).
The discount rate is never deduced here: it belongs to configs/config.yml.

    python -m tools.extract_solution data/raw/data.xlsx

Writes <out> (order,id,period; order empty = unordered set) and <out>.meta.json, which also
records the values the workbook reports next to the solution (maximum value, capacity) so
they can be compared with evaluate-plan.
"""

import argparse
import colorsys
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python tools/extract_solution.py` as well as `python -m`
    sys.path.insert(0, str(ROOT))

from configs import project_path  # noqa: E402
from src.data.dataset import SOLUTION_COLUMNS, Block, Mine, file_hash, integer  # noqa: E402

SOLUTION_SHEET = "Resultado"
BLOCK_HEADERS = ("ID", "X", "Y", "Z")
DEFAULT_OUTPUT = "data/processed/solution.csv"
PERIOD_PATTERN = re.compile(r"periodo\s*(\d+)")
REPORTED_LABELS = {"max_value": "valor maximo", "capacity": "capacidad mina"}


def plain(text: str) -> str:
    """Lowercase without accents, to match legend and label texts written by hand."""
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).lower().split())


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
class Legend:
    cell: str
    text: str
    period: int  # CSV period (0-based)


@dataclass(frozen=True)
class Selection:
    sheet: str
    periods: dict[int, int]  # block id -> CSV period
    legends: tuple[Legend, ...]
    panels: tuple[dict, ...]


def block_sheet_name(workbook, requested: str | None) -> str:
    if requested is not None:
        if requested not in workbook.sheetnames:
            raise ValueError(f"Falta la hoja {requested!r}")
        return requested
    found = [ws.title for ws in workbook.worksheets
             if all(name in next(ws.iter_rows(max_row=1, values_only=True), ()) for name in BLOCK_HEADERS)]
    if len(found) != 1:
        raise ValueError(f"Se esperaba una sola hoja con encabezados {BLOCK_HEADERS}, se encontraron {found}; "
                         "indíquela con --blocks-sheet")
    return found[0]


def read_block_table(workbook, sheet_name: str) -> Mine:
    rows = workbook[sheet_name].iter_rows(values_only=True)
    headers = next(rows, ())
    if any(headers.count(name) != 1 for name in BLOCK_HEADERS):
        raise ValueError(f"Encabezados obligatorios y únicos en {sheet_name!r}: {BLOCK_HEADERS}")
    index = {name: headers.index(name) for name in BLOCK_HEADERS}
    blocks = tuple(
        Block(*(integer(row[index[name]], f"{name} fila {n}") for name in BLOCK_HEADERS), value=0.0, source_row=n)
        for n, row in enumerate(rows, start=2) if not all(v is None for v in row))
    return Mine(blocks)


def read_legends(sheet, workbook) -> dict[tuple, Legend]:
    """Filled empty cells with a text on their right; each colour is one period."""
    found = []
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is not None or cell.fill.patternType != "solid":
                continue
            label = sheet.cell(cell.row, cell.column + 1).value
            if isinstance(label, str) and label.strip():
                found.append((cell, label.strip()))
    if not found:
        raise ValueError(f"{sheet.title}: no hay leyenda (celda pintada vacía con un texto a su derecha)")
    numbered = [PERIOD_PATTERN.search(plain(text)) for _, text in found]
    if any(numbered) and not all(numbered):
        raise ValueError(f"{sheet.title}: algunas leyendas indican período y otras no: {[t for _, t in found]}")
    if not any(numbered) and len(found) > 1:
        raise ValueError(f"{sheet.title}: varias leyendas sin período: {[t for _, t in found]}")
    legends = {}
    for (cell, text), match in zip(found, numbered):
        period = int(match.group(1)) - 1 if match else 0
        if period < 0:
            raise ValueError(f"{sheet.title}!{cell.coordinate}: los períodos del Excel parten en 1")
        signature = fill_signature(cell, workbook)
        if signature in legends:
            raise ValueError(f"{sheet.title}: {cell.coordinate} repite el color de {legends[signature].cell}")
        if period in {legend.period for legend in legends.values()}:
            raise ValueError(f"{sheet.title}: el período {period + 1} aparece en dos leyendas")
        legends[signature] = Legend(cell.coordinate, text, period)
    return legends


def panel_layout(sheet, label_row: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """X header row next to a "Z=" label, its X columns (from C) and its Y rows (column B)."""
    header = label_row if isinstance(sheet.cell(label_row, 3).value, (int, float)) else label_row + 1
    xs, column = [], 3
    while isinstance(sheet.cell(header, column).value, (int, float)):
        xs.append((column, integer(sheet.cell(header, column).value, f"{sheet.title}:X columna {column}")))
        column += 1
    ys, row = [], header + 1
    while isinstance(sheet.cell(row, 2).value, (int, float)):
        ys.append((row, integer(sheet.cell(row, 2).value, f"{sheet.title}:Y fila {row}")))
        row += 1
    if not xs or not ys:
        raise ValueError(f"{sheet.title}: el panel de la fila {label_row} no tiene valores de X (fila) o Y (columna B)")
    return header, xs, ys


def read_selection(workbook, sheet_name: str, mine: Mine) -> Selection:
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Falta la hoja visual {sheet_name}")
    sheet = workbook[sheet_name]
    legends = read_legends(sheet, workbook)
    by_position = {(b.x, b.y, b.z): b.id for b in mine.blocks}
    periods, seen, panels = {}, set(), []
    labels = [(cell.row, cell.value) for cell in sheet["A"] if isinstance(cell.value, str) and cell.value.startswith("Z=")]
    if not labels:
        raise ValueError(f"{sheet_name}: no se encontraron paneles Z=")
    for label_row, text in labels:
        try:
            z = int(text[2:])
        except ValueError as exc:
            raise ValueError(f"Nivel no reconocido: {text}") from exc
        header, xs, ys = panel_layout(sheet, label_row)
        panels.append({"z": z, "header_row": header, "columns": len(xs), "rows": len(ys)})
        for row, y in ys:
            for column, x in xs:
                cell = sheet.cell(row, column)
                block_id = integer(cell.value, f"{sheet_name}!{cell.coordinate}")
                if by_position.get((x, y, z)) != block_id:
                    raise ValueError(f"{sheet_name}!{cell.coordinate}: ID/coordenadas no corresponden a tabla")
                if block_id in seen:
                    raise ValueError(f"Bloque repetido en hoja visual: {block_id}")
                seen.add(block_id)
                if cell.fill.patternType:
                    legend = legends.get(fill_signature(cell, workbook))
                    if legend is not None:
                        periods[block_id] = legend.period
    if seen != set(mine.by_id):
        raise ValueError(f"{sheet_name}: la vista no corresponde al yacimiento completo")
    return Selection(sheet_name, periods, tuple(legends.values()), tuple(panels))


def reported_values(sheet) -> dict:
    """Numbers the workbook writes next to (right of, or below) a known label; informative only."""
    found = {}
    for row in sheet.iter_rows():
        for cell in row:
            if not isinstance(cell.value, str):
                continue
            for key, label in REPORTED_LABELS.items():
                if key in found or not plain(cell.value).startswith(label):
                    continue
                for neighbour in (sheet.cell(cell.row, cell.column + 1), sheet.cell(cell.row + 1, cell.column)):
                    if isinstance(neighbour.value, (int, float)) and not isinstance(neighbour.value, bool):
                        found[key] = {"value": neighbour.value, "cell": neighbour.coordinate, "label": cell.value}
                        break
    return found


def display_path(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def extract(source: Path, output: Path, sheet: str = SOLUTION_SHEET,
            blocks_sheet: str | None = None) -> tuple[Path, Path]:
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    workbook = load_workbook(source, data_only=True)
    try:
        table = block_sheet_name(workbook, blocks_sheet)
        selection = read_selection(workbook, sheet, read_block_table(workbook, table))
        reported = reported_values(workbook[sheet])
    finally:
        workbook.close()
    if not selection.periods:
        raise ValueError(f"{sheet}: no hay celdas con el color de una leyenda")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(selection.periods.items(), key=lambda item: (item[1], item[0]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOLUTION_COLUMNS)
        writer.writerows(("", block_id, period) for block_id, period in rows)
    counts = {}
    for _, period in rows:
        counts[period] = counts.get(period, 0) + 1
    meta_path = output.with_suffix(".meta.json")
    meta = {
        "schema": "openpit-solution", "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/extract_solution.py",
        "source": {"path": display_path(source), "sha256": file_hash(source), "sheet": sheet,
                   "blocks_sheet": table, "panels": list(selection.panels),
                   "legends": [{"cell": l.cell, "text": l.text, "period": l.period} for l in selection.legends]},
        "ordered": False,
        "period_note": "Período N del Excel = period N-1 del CSV",
        "blocks_per_period": {str(p): counts[p] for p in sorted(counts)},
        "reported": reported,
        "reported_note": "Valores escritos en el Excel junto a la solución; solo informativos. "
                         "La tasa de descuento no se deduce: se fija en configs/config.yml.",
        "output": {"path": display_path(output), "sha256": file_hash(output), "rows": len(rows)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, meta_path


def main(argv=None):
    cli = argparse.ArgumentParser(description="Solución pintada en un Excel -> CSV estándar de solución")
    cli.add_argument("source", help="Excel con la solución; relativo al repositorio o absoluto")
    cli.add_argument("--sheet", default=SOLUTION_SHEET, help=f"Hoja con la solución pintada; default={SOLUTION_SHEET}")
    cli.add_argument("--blocks-sheet", help="Hoja con la tabla de bloques (ID, X, Y, Z); por defecto la única que la tenga")
    cli.add_argument("--out", default=DEFAULT_OUTPUT, help=f"CSV de salida; default={DEFAULT_OUTPUT}")
    args = cli.parse_args(argv)
    try:
        output, meta = extract(project_path(args.source), project_path(args.out), args.sheet, args.blocks_sheet)
    except (ValueError, OSError, KeyError) as exc:
        cli.exit(2, f"Error: {exc}\n")
    print(f"Solución: {output}\nMetadatos: {meta}")


if __name__ == "__main__":
    main()
