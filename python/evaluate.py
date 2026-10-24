"""Accuracy of the exported Q8.8 model through the bit-exact integer path.

    python python/evaluate.py                    # software oracle (full_inference)
    python python/evaluate.py --port COM4        # also run attention on the FPGA

The task is reconstruction: the model maps a noisy grid to the clean shifted
pattern. A pixel is lit when its sigmoid output is at least 0.5 (128 in Q8.8).
The class is the nearest clean template; "cross" and "plus" in dataset.py are
the same pattern, so they are scored as one class.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).parent))
import dataset
from model_export import QuantModel, full_inference, load_model

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 128
SAME_PATTERN = {"plus": "cross"}

Grid = list[list[int]]


def binarize(output: Sequence[Sequence[int]]) -> Grid:
    return [[int(value >= THRESHOLD) for value in row] for row in output]


def class_of(label: str) -> str:
    return SAME_PATTERN.get(label, label)


def _templates() -> list[tuple[str, Grid]]:
    return [(class_of(name), dataset.shifted(image, dr, dc))
            for name, image in dataset.patterns().items() if name not in SAME_PATTERN
            for dr in range(-2, 3) for dc in range(-2, 3)]


TEMPLATES = _templates()


def classify(grid: Grid) -> str:
    def distance(template: Grid) -> int:
        return sum(a != b for ra, rb in zip(grid, template) for a, b in zip(ra, rb))
    return min(TEMPLATES, key=lambda item: distance(item[1]))[0]


def score(records: Sequence[dict], predictions: Sequence[Grid]) -> dict[str, float]:
    pixels = exact = classes = baseline = 0
    for record, predicted in zip(records, predictions):
        target = record["target"]
        wrong = sum(p != t for rp, rt in zip(predicted, target) for p, t in zip(rp, rt))
        pixels += 256 - wrong
        exact += wrong == 0
        classes += classify(predicted) == class_of(record["label"])
        baseline += sum(p == t for rp, rt in zip(record["pixels"], target) for p, t in zip(rp, rt))
    n = len(records)
    return {"samples": n, "pixel_accuracy": pixels / (256 * n), "exact_grids": exact / n,
            "class_accuracy": classes / n, "input_pixel_accuracy": baseline / (256 * n)}


def held_out(path: Path = ROOT / "data" / "patterns.json") -> dict[str, list[dict]]:
    if not path.exists():
        dataset.build(path)
    return {"test": dataset.split(dataset.load(path))[1], "hard": dataset.hard_set()}


def run(model: QuantModel, records: Sequence[dict], transport: Callable | None = None) -> list[Grid]:
    return [full_inference(r["pixels"], model.attention, model.feed_forward, model.output,
                           attention_transport=transport) for r in records]


def report(name: str, metrics: dict[str, float]) -> None:
    print(f"{name}: {metrics['samples']} samples, pixel {metrics['pixel_accuracy']:.2%} "
          f"(input {metrics['input_pixel_accuracy']:.2%}), exact grids {metrics['exact_grids']:.2%}, "
          f"class {metrics['class_accuracy']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--port", help="also run attention on the FPGA at this UART port")
    parser.add_argument("--limit", type=int, default=0, help="samples per set (0 = all)")
    args = parser.parse_args()
    model = load_model(args.weights)
    sets = {name: records[:args.limit or None] for name, records in held_out().items()}
    tpu = None
    if args.port:
        from host import TpuLink
        tpu = TpuLink(args.port)
        for layer, weights in enumerate(model.attention):
            tpu.load_weights(layer, weights)
    ok = True
    try:
        for name, records in sets.items():
            start = time.perf_counter()
            oracle = run(model, records)
            print(f"[{name}] oracle: {time.perf_counter() - start:.0f} s")
            report(f"{name} (oracle)", score(records, [binarize(o) for o in oracle]))
            if tpu is None:
                continue
            start = time.perf_counter()
            fpga = run(model, records, tpu.attention)
            same = sum(f == o for f, o in zip(fpga, oracle))
            print(f"[{name}] fpga: {time.perf_counter() - start:.0f} s, "
                  f"{same}/{len(records)} outputs identical to the oracle")
            report(f"{name} (fpga)", score(records, [binarize(f) for f in fpga]))
            ok = ok and same == len(records)
    finally:
        if tpu is not None:
            tpu.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
