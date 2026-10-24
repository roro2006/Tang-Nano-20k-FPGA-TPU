"""Build the UART byte script for sim/tb_tpu_top.sv from the Python oracle.

Each output line is three hex digits: 0xx drives byte xx into the FPGA, 1xx
expects byte xx back, and fff ends the script.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from host import (OP_ATTN, OP_ECHO, OP_ERROR, OP_INFO, OP_LOAD, attention_request,
                  encode_q8_matrix, load_request, random_layer, random_matrix,
                  request, response)
from q8_attention import attention_q8_8, dot_q8_8, matmul_q8_8, sat16

ROOT = Path(__file__).resolve().parents[1]


def peaks_at_last_key(x, weights) -> bool:
    """True when some row in each 8-row group has its first score maximum at key 15."""
    q, k = matmul_q8_8(x, weights["q"]), matmul_q8_8(x, weights["k"])
    scores = [[sat16(dot_q8_8(qr, kr) >> 3) for kr in k] for qr in q]
    last = [row.index(max(row)) == 15 for row in scores]
    return any(last[:8]) and any(last[8:])


def build(seed: int = 5) -> list[tuple[bytes, bytes]]:
    rng = random.Random(seed)
    steps = [(request(OP_INFO), response(OP_INFO, 0, b"TPU\x01")),
             (request(OP_ECHO, 7, b"hello"), response(OP_ECHO, 0, b"hello"))]
    corrupt = bytearray(request(OP_ECHO, 0, b"xyz"))
    corrupt[-1] ^= 0xFF
    steps.append((bytes(corrupt), response(OP_ERROR, 1)))

    layer = 1
    weights = random_layer(rng)
    for name in ("q", "k", "v", "o"):
        steps.append((load_request(layer, name, weights[name]), response(OP_LOAD)))
    # A moderate input whose rows peak at the last key in both row groups (the
    # softmax write/fix-up boundary), then a large one that saturates.
    for span in (512, 6000):
        x = random_matrix(rng, 16, 64, span)
        while span == 512 and not peaks_at_last_key(x, weights):
            x = random_matrix(rng, 16, 64, span)
        expected = attention_q8_8(x, weights["q"], weights["k"], weights["v"], weights["o"])
        steps.append((attention_request(layer, x), response(OP_ATTN, 0, encode_q8_matrix(expected))))
    steps.append((request(OP_ATTN, layer, b"\x00" * 10), response(OP_ATTN, 1)))
    return steps


def main() -> None:
    lines = []
    for sent, expected in build():
        lines += [f"0{b:02x}" for b in sent] + [f"1{b:02x}" for b in expected]
    lines.append("fff")
    out = ROOT / "sim" / "tpu_top_vectors.mem"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} entries)")


if __name__ == "__main__":
    main()
