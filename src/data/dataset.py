"""Read the pipeline's standard CSV files into immutable objects.

The pipeline only reads two standard formats (see tools/README.md):
- blocks:   id,x,y,z,value[,tonnes]   (higher z = shallower; the surface is the maximum z)
- plan:     order,id,period            (order empty in every row = unordered set)
Other source formats (Excel, etc.) are converted beforehand by the scripts in tools/.
"""

import csv
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


BLOCK_COLUMNS = ("id", "x", "y", "z", "value")
OPTIONAL_BLOCK_COLUMNS = ("tonnes",)
SOLUTION_COLUMNS = ("order", "id", "period")


def number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field}: se requiere un número finito, recibido {value!r}")
    return float(value)


def integer(value, field: str) -> int:
    numeric = number(value, field)
    if not numeric.is_integer():
        raise ValueError(f"{field}: se requiere un entero")
    return int(numeric)


def parse_number(text: str, field: str) -> float:
    """CSV cells are text: convert explicitly and reject empty or non-numeric cells."""
    try:
        return number(float(text), field)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: se requiere un número finito, recibido {text!r}") from None


def parse_integer(text: str, field: str) -> int:
    return integer(parse_number(text, field), field)


@dataclass(frozen=True)
class Block:
    id: int
    x: int
    y: int
    z: int
    value: float
    source_row: int | None = None
    tonnes: float | None = None
    reachable: bool = True


@dataclass(frozen=True)
class Mine:
    blocks: tuple[Block, ...]
    source: str = "synthetic"
    data_sha256: str | None = None

    def __post_init__(self):
        if not self.blocks:
            raise ValueError("El yacimiento no contiene bloques")
        ids, coordinates = set(), set()
        for block in self.blocks:
            if isinstance(block.id, bool) or not isinstance(block.id, int):
                raise ValueError("ID interno debe ser int")
            position = tuple(integer(getattr(block, c), c) for c in ("x", "y", "z"))
            number(block.value, "value")
            if block.id in ids or position in coordinates:
                raise ValueError("ID o coordenadas duplicados")
            if block.tonnes is not None and number(block.tonnes, "tonnes") <= 0:
                raise ValueError("tonnes debe ser positivo")
            ids.add(block.id)
            coordinates.add(position)

    @property
    def by_id(self) -> Mapping[int, Block]:
        return MappingProxyType({block.id: block for block in self.blocks})


@dataclass(frozen=True)
class SolutionEntry:
    block_id: int
    period: int
    order: int | None = None


@dataclass(frozen=True)
class Solution:
    """A reference solution. `ordered` is False when the source gives only a set of blocks."""
    entries: tuple[SolutionEntry, ...]
    ordered: bool
    source: str = "synthetic"
    data_sha256: str | None = None

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(entry.block_id for entry in self.entries)


def file_hash(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path: Path, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> list[tuple[int, dict]]:
    path = Path(path).resolve(strict=True)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError(f"{path.name}: encabezados repetidos {headers}")
        missing = [c for c in required if c not in headers]
        unknown = [c for c in headers if c not in required + optional]
        if missing or unknown:
            raise ValueError(f"{path.name}: columnas obligatorias {required}, opcionales {optional}; "
                             f"faltan {missing}, desconocidas {unknown}")
        return [(row_number, row) for row_number, row in enumerate(reader, start=2)
                if any((value or "").strip() for value in row.values())]


def reachable_ids(blocks) -> frozenset[int]:
    """Pit rule (45° slope with cubic blocks): a block is reachable only if its whole
    overlying cone exists in the grid: 3x3 blocks one level up, 5x5 two levels up, ...
    up to the surface (maximum z). Blocks on the edges of the model are unreachable.
    """
    positions = {(b.x, b.y, b.z) for b in blocks}
    surface_z = max(b.z for b in blocks)
    return frozenset(b.id for b in blocks if all(
        (b.x + dx, b.y + dy, b.z + distance) in positions
        for distance in range(1, surface_z - b.z + 1)
        for dx in range(-distance, distance + 1) for dy in range(-distance, distance + 1)))


def load_mine(path: Path) -> Mine:
    """Standard block CSV -> Mine with `reachable` computed from the pit rule.

    The hash identifies the exact CSV used in an experiment.
    """
    path = Path(path).resolve(strict=True)
    blocks = []
    for row_number, row in _read_csv(path, BLOCK_COLUMNS, OPTIONAL_BLOCK_COLUMNS):
        tonnes = (row.get("tonnes") or "").strip()
        blocks.append(Block(
            id=parse_integer(row["id"], f"id fila {row_number}"),
            x=parse_integer(row["x"], f"x fila {row_number}"),
            y=parse_integer(row["y"], f"y fila {row_number}"),
            z=parse_integer(row["z"], f"z fila {row_number}"),
            value=parse_number(row["value"], f"value fila {row_number}"),
            source_row=row_number,
            tonnes=parse_number(tonnes, f"tonnes fila {row_number}") if tonnes else None,
        ))
    reachable = reachable_ids(blocks) if blocks else frozenset()
    blocks = [replace(b, reachable=b.id in reachable) for b in blocks]
    return Mine(tuple(blocks), str(path), file_hash(path))


def load_solution(path: Path, mine: Mine) -> Solution:
    """Standard plan CSV (order,id,period) -> Solution, checked against the mine's block IDs."""
    path = Path(path).resolve(strict=True)
    entries = []
    for row_number, row in _read_csv(path, SOLUTION_COLUMNS):
        order = (row["order"] or "").strip()
        entries.append(SolutionEntry(
            block_id=parse_integer(row["id"], f"id fila {row_number}"),
            period=parse_integer(row["period"], f"period fila {row_number}"),
            order=parse_integer(order, f"order fila {row_number}") if order else None,
        ))
    if not entries:
        raise ValueError(f"{path.name}: la solución no contiene bloques")
    ids = [entry.block_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: bloques repetidos en la solución")
    unknown = sorted(set(ids) - set(mine.by_id))
    if unknown:
        raise ValueError(f"{path.name}: bloques que no existen en los datos: {unknown[:10]}")
    if any(entry.period < 0 for entry in entries):
        raise ValueError(f"{path.name}: period debe ser >= 0")
    orders = [entry.order for entry in entries]
    if all(order is None for order in orders):
        ordered = False
    elif sorted(orders, key=lambda o: (o is None, o)) == list(range(1, len(entries) + 1)):
        ordered = True
        entries.sort(key=lambda entry: entry.order)
    else:
        raise ValueError(f"{path.name}: order debe estar vacío en todas las filas o ser 1..{len(entries)} sin repetir")
    return Solution(tuple(entries), ordered, str(path), file_hash(path))
