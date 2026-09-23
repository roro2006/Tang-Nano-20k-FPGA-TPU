"""Generate softmax ROMs and deterministic attention vectors.

Usage: python3 python/generate_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from q8_attention import EXP_LUT, attention_q8_8, q8_8

ROOT = Path(__file__).resolve().parents[1]


def write_mem(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{value & 0xffff:04x}\n" for value in values))


def main() -> None:
    (ROOT / "weights").mkdir(exist_ok=True)
    write_mem(ROOT / "weights" / "softmax_exp.mem", list(EXP_LUT))
    reciprocal = [0 if n == 0 else min(0xffff, round((1 << 24) / n))
                  for n in range(65536)]
    write_mem(ROOT / "weights" / "softmax_recip.mem", reciprocal)
    eye = [[256 if row == col else 0 for col in range(64)] for row in range(64)]
    x = [[q8_8(((row + col) % 9 - 4) / 8) for col in range(64)] for row in range(16)]
    output = attention_q8_8(x, eye, eye, eye)
    (ROOT / "sim").mkdir(exist_ok=True)
    (ROOT / "sim" / "attention_vector.json").write_text(json.dumps({"x": x, "expected": output}))
    print("generated weights/softmax_exp.mem and sim/attention_vector.json")


if __name__ == "__main__":
    main()
