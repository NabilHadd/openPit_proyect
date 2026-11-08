"""Dependency-free SVG learning curve: VAN of each training episode and best so far."""

from pathlib import Path


def plot_history(values: list[float], path: Path):
    if not values:
        return
    best, running = [], float("-inf")
    for value in values:
        running = max(running, value)
        best.append(running)
    low, high = min(values), max(values)
    scale = high - low or 1

    def points(series):
        return " ".join(f"{40 + i * 700 / max(1, len(series) - 1):.2f},{250 - (v - low) * 200 / scale:.2f}"
                        for i, v in enumerate(series))
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300" viewBox="0 0 800 300">'
        '<rect width="800" height="300" fill="white"/>'
        '<text x="40" y="20">VAN por episodio (gris) y mejor episodio hasta el momento (azul)</text>'
        f'<text x="40" y="285">episodios: {len(values)}; mín: {low:g}; máx: {high:g}</text>'
        f'<polyline points="{points(values)}" fill="none" stroke="#bbb" stroke-width="1"/>'
        f'<polyline points="{points(best)}" fill="none" stroke="#246" stroke-width="2"/></svg>\n',
        encoding="utf-8")
