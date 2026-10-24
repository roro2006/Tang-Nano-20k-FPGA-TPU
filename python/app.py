"""Tkinter 16x16 drawing front end for the trained pattern denoiser.

    python python/app.py COM4          # attention on the FPGA
    python python/app.py --oracle      # software oracle, no board needed

Draw on the left grid, press Infer, and the right grid shows the model's
reconstruction plus the nearest pattern class. The attention weights are
loaded into the FPGA once at start-up; the rest of the model runs here.
"""
from __future__ import annotations
import argparse
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import binarize, classify
from host import TpuLink
from model_export import QuantModel, full_inference, load_model

CELL = 20
WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


class DrawingApp:
    def __init__(self, model: QuantModel, tpu: TpuLink | None):
        self.model, self.tpu = model, tpu
        self.grid = [[0] * 16 for _ in range(16)]
        self.root = tk.Tk()
        self.root.title(f"TPU denoiser ({'FPGA' if tpu else 'software oracle'})")
        self.canvas = tk.Canvas(self.root, width=16 * CELL * 2 + CELL, height=16 * CELL)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.status = tk.Label(self.root, text="draw a cross, box or x")
        self.status.pack()
        tk.Button(self.root, text="Infer", command=self.infer).pack(side=tk.LEFT)
        tk.Button(self.root, text="Clear", command=self.clear).pack(side=tk.LEFT)
        self.cells = self._cells(0)
        self.outputs = self._cells(17 * CELL)

    def _cells(self, left: int):
        return [[self.canvas.create_rectangle(left + c * CELL, r * CELL, left + c * CELL + CELL,
                 r * CELL + CELL, fill="white") for c in range(16)] for r in range(16)]

    def paint(self, event):
        r, c = event.y // CELL, event.x // CELL
        if 0 <= r < 16 and 0 <= c < 16:
            self.grid[r][c] = 1
            self.canvas.itemconfigure(self.cells[r][c], fill="black")

    def clear(self):
        self.grid = [[0] * 16 for _ in range(16)]
        for row in self.cells + self.outputs:
            for cell in row:
                self.canvas.itemconfigure(cell, fill="white")

    def infer(self):
        transport = self.tpu.attention if self.tpu else None
        output = binarize(full_inference(self.grid, self.model.attention, self.model.feed_forward,
                                         self.model.output, attention_transport=transport))
        for r in range(16):
            for c in range(16):
                self.canvas.itemconfigure(self.outputs[r][c], fill="black" if output[r][c] else "white")
        self.status.configure(text=f"looks like: {classify(output)}")

    def run(self):
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", nargs="?", help="UART port, e.g. COM4")
    parser.add_argument("--oracle", action="store_true", help="run attention in software")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if not args.port and not args.oracle:
        parser.error("give a port or --oracle")
    model = load_model(args.weights)
    if args.oracle:
        DrawingApp(model, None).run()
        return
    with TpuLink(args.port) as tpu:
        for layer, weights in enumerate(model.attention):
            tpu.load_weights(layer, weights)
        DrawingApp(model, tpu).run()


if __name__ == "__main__":
    main()
