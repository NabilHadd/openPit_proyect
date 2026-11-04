"""tools/prepare_blocks.py: any CSV/Excel table -> the pipeline's standard blocks CSV."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.dataset import file_hash, load_mine
from tools.prepare_blocks import parse_column_map, prepare, standardize


class PrepareBlocksTests(unittest.TestCase):
    def test_excel_with_custom_names_sheet_and_extra_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.xlsx", root / "processed" / "data.csv"
            table = pd.DataFrame({"BlockID": [1, 2], "E": [1, 2], "N": [1, 1], "Cota": [3, 3],
                                  "Valor": [5.5, -2], "Ley": [0.1, 0.2]})
            with pd.ExcelWriter(source) as writer:
                pd.DataFrame({"otra": [1]}).to_excel(writer, sheet_name="Portada", index=False)
                table.to_excel(writer, sheet_name="Bloques", index=False)
            mapping = parse_column_map(["BlockID=id", "E=x", "N=y", "Cota=z", "Valor=value"])
            data, meta_path = prepare(source, output, mapping, "z-max", sheet="Bloques")
            mine = load_mine(data)
            self.assertEqual([(b.id, b.x, b.value) for b in mine.blocks], [(1, 1, 5.5), (2, 2, -2.0)])
            meta = json.loads(meta_path.read_text())
            self.assertEqual(meta_path.name, "data.meta.json")
            self.assertEqual(meta["source"]["sha256"], file_hash(source))
            self.assertEqual(meta["output"]["sha256"], mine.data_sha256)
            self.assertEqual(meta["dropped_columns"], ["Ley"])

    def test_csv_already_standard_with_optional_tonnes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.csv", Path(directory) / "data.csv"
            source.write_text("id,x,y,z,value,tonnes\n1,1,1,1,3,10\n2,2,1,1,4,\n")
            prepare(source, output, {}, "z-max")
            self.assertEqual([b.tonnes for b in load_mine(output).blocks], [10.0, None])

    def test_surface_z_min_is_inverted_to_pipeline_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.csv", Path(directory) / "data.csv"
            source.write_text("id,x,y,z,value\n1,1,1,1,-1\n2,1,1,2,5\n3,1,1,4,9\n")  # z=1 is the surface
            _, meta_path = prepare(source, output, {}, "z-min")
            self.assertEqual({b.id: b.z for b in load_mine(output).blocks}, {1: 4, 2: 3, 3: 1})
            self.assertTrue(json.loads(meta_path.read_text())["z_inverted"])
            prepare(source, output, {}, "z-max")
            self.assertEqual({b.id: b.z for b in load_mine(output).blocks}, {1: 1, 2: 2, 3: 4})
            with self.assertRaises(ValueError):
                prepare(source, output, {}, "up")

    def test_invalid_rows_are_reported_not_repaired(self):
        cases = (
            ({"id": [1, 1], "x": [1, 2], "y": [1, 1], "z": [1, 1], "value": [1, 2]}, "id repetidos"),
            ({"id": [1, 2], "x": [1, 1], "y": [1, 1], "z": [1, 1], "value": [1, 2]}, "x/y/z repetidos"),
            ({"id": [1, 2], "x": [1, 2.5], "y": [1, 1], "z": [1, 1], "value": [1, 2]}, "x: valores inválidos"),
            ({"id": [1, 2], "x": [1, 2], "y": [1, 1], "z": [1, 1], "value": [1, None]}, "value: valores inválidos"),
            ({"id": [1, 2], "x": [1, 2], "y": [1, 1], "z": [1, 1], "value": [1, "abc"]}, "fila 3"),
            ({"id": [1, 2], "x": [1, 2], "y": [1, 1], "z": [1, 1], "value": [True, 2]}, "value: valores inválidos"),
            ({"id": [1], "x": [1], "y": [1], "z": [1], "value": [1], "tonnes": [-1]}, "tonnes"),
            ({"id": [1], "x": [1], "y": [1], "z": [1]}, "Faltan columnas obligatorias"),
        )
        for columns, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                standardize(pd.DataFrame(columns), {})
        with self.assertRaisesRegex(ValueError, "inexistentes"):
            standardize(pd.DataFrame({"id": [1]}), {"ID": "id"})
        for items in (["ID"], ["ID=identificador"], ["A=id", "B=id"]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                parse_column_map(items)


if __name__ == "__main__":
    unittest.main()
