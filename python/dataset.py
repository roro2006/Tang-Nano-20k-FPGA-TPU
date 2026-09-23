"""Deterministic 16x16 pattern dataset generator."""
from __future__ import annotations
import random
from pathlib import Path
import json

def patterns() -> dict[str, list[list[int]]]:
    out = {}
    for name, fn in {
        "cross": lambda r, c: r == 7 or r == 8 or c == 7 or c == 8,
        "box": lambda r, c: r in (2, 13) or c in (2, 13),
        "x": lambda r, c: r == c or r + c == 15,
        "plus": lambda r, c: r in (7, 8) or c in (7, 8),
    }.items():
        out[name] = [[int(fn(r, c)) for c in range(16)] for r in range(16)]
    return out

def build(path: Path, copies: int = 300, seed: int = 7) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = patterns()
    records = []
    for name, image in base.items():
        for _ in range(copies):
            dr, dc = rng.randint(-2, 2), rng.randint(-2, 2)
            pixels = [[0] * 16 for _ in range(16)]
            for r in range(16):
                for c in range(16):
                    rr, cc = r - dr, c - dc
                    value = image[rr][cc] if 0 <= rr < 16 and 0 <= cc < 16 else 0
                    pixels[r][c] = int(value and rng.random() > 0.03)
            records.append({"label": name, "pixels": pixels})
    path.write_text(json.dumps(records))

if __name__ == "__main__":
    build(Path("data/patterns.json"))
