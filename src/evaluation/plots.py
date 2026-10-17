"""Optional dependency-free local SVG: episode NPV, not a mining-plan layout."""

from pathlib import Path


def plot_history(history, path: Path):
    values = [record.evaluation.npv for record in history]
    if not values:
        return
    low, high = min(values), max(values)
    scale = high - low or 1
    points = " ".join(f"{40 + i * 700 / max(1, len(values) - 1):.2f},{250 - (v - low) * 200 / scale:.2f}"
                      for i, v in enumerate(values))
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300">'
        '<rect width="800" height="300" fill="white"/>'
        '<text x="40" y="20">VAN recalculado por episodio (no mejor acumulado)</text>'
        f'<text x="40" y="285">episodios: {len(values)}; min: {low:g}; max: {high:g}</text>'
        f'<polyline points="{points}" fill="none" stroke="#246" stroke-width="1"/></svg>\n',
        encoding="utf-8",
    )
