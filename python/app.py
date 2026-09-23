"""Tkinter 16x16 drawing front end and PC-side inference loop."""
from __future__ import annotations
import tkinter as tk
from q8_attention import q8_8
from host import encode_q8_matrix, run_once, decode_q8_matrix

class DrawingApp:
    def __init__(self, port: str):
        self.port, self.grid = port, [[0] * 16 for _ in range(16)]
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=320, height=320)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        tk.Button(self.root, text="Infer", command=self.infer).pack()
        self.cells = [[self.canvas.create_rectangle(c * 20, r * 20, c * 20 + 20,
                       r * 20 + 20, fill="white") for c in range(16)] for r in range(16)]

    def paint(self, event):
        r, c = min(15, event.y // 20), min(15, event.x // 20)
        self.grid[r][c] = 1
        self.canvas.itemconfigure(self.cells[r][c], fill="black")

    def embed(self):
        return [[q8_8(self.grid[r][c]) for c in range(16)] +
                [q8_8(1.0 if r == c else 0.0) for c in range(16, 64)]
                for r in range(16)]

    def infer(self):
        response = run_once(self.port, 115200, encode_q8_matrix(self.embed()), 5.0)
        output = decode_q8_matrix(response, 16, 64)
        for r in range(16):
            for c in range(16):
                self.canvas.itemconfigure(self.cells[r][c], fill="black" if output[r][c] > 0 else "white")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    import sys
    DrawingApp(sys.argv[1]).run()
