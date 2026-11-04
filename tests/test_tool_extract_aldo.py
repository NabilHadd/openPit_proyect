"""Colour handling of tools/extract_aldo_solution.py (Aldo's workbook, outside the pipeline)."""

import unittest

from openpyxl import Workbook
from openpyxl.styles import Color, PatternFill

from tools.extract_aldo_solution import color_rgb, fill_signature


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


if __name__ == "__main__":
    unittest.main()
