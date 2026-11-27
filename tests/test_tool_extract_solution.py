"""tools/extract_solution.py: colours, layout detection and periods (outside the pipeline)."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from openpyxl import Workbook
from openpyxl.styles import Color, PatternFill

from tools.extract_solution import color_rgb, extract, fill_signature

PERIOD_1 = PatternFill("solid", fgColor="004BACC6")
PERIOD_2 = PatternFill("solid", fgColor="00F79646")
OTHER = PatternFill("solid", fgColor="00FFFF00")  # painted, but no legend uses it


def workbook_file(folder: Path, nx: int, ny: int, nz: int, painted: dict, legends: list,
                  blocks_sheet: str = "Bloques", label_on_first_row: bool = True,
                  reported: dict | None = None) -> Path:
    """A workbook laid out like the client's: block table + one X-Y panel per level (top level first).

    painted: block id -> fill. legends: (cell, fill, text). reported: (label cell, value cell) -> (label, value)."""
    workbook = Workbook()
    table = workbook.active
    table.title = blocks_sheet
    table.append(["ID", "X", "Y", "Z", "Valor"])
    ids = {}
    for z in range(1, nz + 1):
        for y in range(1, ny + 1):
            for x in range(1, nx + 1):
                ids[x, y, z] = len(ids) + 1
                table.append([ids[x, y, z], x, y, z, 0])
    sheet = workbook.create_sheet("Resultado")
    row = 1
    for z in range(nz, 0, -1):
        if row == 1 and label_on_first_row:   # first panel: label above the X header
            sheet.cell(row, 1, f"Z={z}")
            sheet.cell(row, 3, "X")
            header = row + 1
        else:                                  # later panels: label on the X header row
            sheet.cell(row, 3, "X")
            header = row + 1
            sheet.cell(header, 1, f"Z={z}")
        for x in range(1, nx + 1):
            sheet.cell(header, 2 + x, x)
        for offset, y in enumerate(range(ny, 0, -1), start=1):
            sheet.cell(header + offset, 2, y)
            for x in range(1, nx + 1):
                cell = sheet.cell(header + offset, 2 + x, ids[x, y, z])
                if ids[x, y, z] in painted:
                    cell.fill = painted[ids[x, y, z]]
        row = header + ny + 2
    for coordinate, fill, text in legends:
        sheet[coordinate].fill = fill
        sheet[coordinate].offset(0, 1).value = text
    for (label_cell, value_cell), (label, value) in (reported or {}).items():
        sheet[label_cell] = label
        sheet[value_cell] = value
    path = folder / "book.xlsx"
    workbook.save(path)
    return path


def read_rows(path: Path) -> list[tuple[str, int, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(r["order"], int(r["id"]), int(r["period"])) for r in csv.DictReader(handle)]


class ColourTests(unittest.TestCase):
    def test_theme_and_rgb_fill(self):
        from openpyxl.writer.theme import theme_xml
        workbook = Workbook()
        workbook.loaded_theme = theme_xml
        a, b = workbook.active["A1"], workbook.active["B1"]
        a.fill = PatternFill("solid", fgColor=Color(theme=8))
        b.fill = PatternFill("solid", fgColor="004BACC6")
        self.assertEqual(fill_signature(a, workbook), fill_signature(b, workbook))

    def test_tint_uses_luminance_and_indexed_palette(self):
        workbook = Workbook()
        # RGB (0.2, 0.4, 0.6): HLS lightness 0.4 -> 0.7, saturation unchanged.
        self.assertEqual(color_rgb(Color(rgb="00336699", tint=0.5), workbook), "8CB2D9")
        self.assertEqual(color_rgb(Color(rgb="00336699", tint=1), workbook), "FFFFFF")
        self.assertEqual(color_rgb(Color(rgb="00336699", tint=-1), workbook), "000000")
        self.assertEqual(color_rgb(Color(indexed=5), workbook), "FFFF00")


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.out = self.folder / "solution.csv"

    def run_extract(self, path: Path, **kwargs):
        extract(path, self.out, **kwargs)
        return read_rows(self.out), json.loads(self.out.with_suffix(".meta.json").read_text(encoding="utf-8"))

    def test_single_legend_without_period_goes_to_period_zero(self):
        path = workbook_file(self.folder, 3, 2, 2, {4: PERIOD_1, 1: PERIOD_1, 2: OTHER},
                             [("K3", PERIOD_1, "Bloques considerados en la solución")],
                             reported={("K13", "L13"): ("Valor Máximo", 336000)})
        rows, meta = self.run_extract(path)
        self.assertEqual(rows, [("", 1, 0), ("", 4, 0)])  # OTHER is painted but not in the legend
        self.assertEqual(meta["source"]["blocks_sheet"], "Bloques")
        self.assertEqual(meta["reported"]["max_value"]["value"], 336000)
        self.assertNotIn("capacity", meta["reported"])

    def test_period_legends_any_grid_size(self):
        # 5 x 4 x 3 grid (not the 7 x 5 of the first workbook); Período N -> period N-1
        painted = {1: PERIOD_1, 2: PERIOD_1, 21: PERIOD_2, 60: PERIOD_2, 30: OTHER}
        path = workbook_file(self.folder, 5, 4, 3, painted,
                             [("O3", PERIOD_1, "Bloques extraídos Período 1"),
                              ("O4", PERIOD_2, "Bloques extraidos periodo 2")],
                             blocks_sheet="60 Bloques", label_on_first_row=False,
                             reported={("O8", "O9"): ("Valor Máximo", 705436.5), ("O11", "P11"): ("Capacidad Mina", 108)})
        rows, meta = self.run_extract(path)
        self.assertEqual(rows, [("", 1, 0), ("", 2, 0), ("", 21, 1), ("", 60, 1)])
        self.assertEqual(meta["blocks_per_period"], {"0": 2, "1": 2})
        self.assertEqual([p["columns"] for p in meta["source"]["panels"]], [5, 5, 5])
        self.assertEqual([p["rows"] for p in meta["source"]["panels"]], [4, 4, 4])
        self.assertEqual(meta["reported"]["max_value"]["cell"], "O9")   # value below its label
        self.assertEqual(meta["reported"]["capacity"]["value"], 108)    # value right of its label

    def test_blocks_sheet_can_be_named(self):
        path = workbook_file(self.folder, 2, 2, 1, {1: PERIOD_1}, [("K3", PERIOD_1, "Solución")], blocks_sheet="Tabla")
        rows, _ = self.run_extract(path, blocks_sheet="Tabla")
        self.assertEqual(rows, [("", 1, 0)])
        with self.assertRaisesRegex(ValueError, "Falta la hoja"):
            extract(path, self.out, blocks_sheet="No existe")

    def test_invalid_legends_are_rejected(self):
        cases = {
            "algunas leyendas indican período": [("O3", PERIOD_1, "Período 1"), ("O4", PERIOD_2, "Solución")],
            "varias leyendas sin período": [("O3", PERIOD_1, "Solución"), ("O4", PERIOD_2, "Otra")],
            "repite el color": [("O3", PERIOD_1, "Período 1"), ("O4", PERIOD_1, "Período 2")],
            "aparece en dos leyendas": [("O3", PERIOD_1, "Período 1"), ("O4", PERIOD_2, "Período 1")],
            "parten en 1": [("O3", PERIOD_1, "Período 0")],
            "no hay leyenda": [],
        }
        for message, legends in cases.items():
            with self.subTest(message), tempfile.TemporaryDirectory() as folder:
                path = workbook_file(Path(folder), 2, 2, 1, {1: PERIOD_1}, legends)
                with self.assertRaisesRegex(ValueError, message):
                    extract(path, Path(folder) / "s.csv")

    def test_nothing_painted_with_a_legend_colour(self):
        path = workbook_file(self.folder, 2, 2, 1, {1: OTHER}, [("K3", PERIOD_1, "Solución")])
        with self.assertRaisesRegex(ValueError, "no hay celdas con el color"):
            extract(path, self.out)


if __name__ == "__main__":
    unittest.main()
