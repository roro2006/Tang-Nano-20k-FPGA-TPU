# Tang Nano 20K FPGA-TPU

This project is a hardware/software starting point for the 16-token, 64-wide
attention pipeline described in the design brief. The FPGA owns the attention
weights and attention arithmetic; the PC owns embedding, residual/layer norm,
feed-forward, and output projection.

The RTL uses an **8×4 systolic array** (32 registered multiply-accumulate
cells), which fits within the Tang Nano 20K's 48 DSP blocks. A tile is fed
through staggered A-row and B-column edge streams, then drains in 74 cycles
(64 terms plus the 8+4-2 pipeline span). Matrices are tiled as 8 output rows
by 4 output columns. The Python implementation in `python/q8_attention.py`
is the bit-exact reference for signed Q8.8 products and rounding.

## Layout

* `python/q8_attention.py` — integer Q8.8 attention oracle and packet codec.
* `rtl/pe.sv` — one registered processing element.
* `rtl/systolic_8x4.sv` — 8×4 tiled MAC array interface.
* `rtl/uart_rx.sv`, `rtl/uart_tx.sv` — 115200 baud, 27 MHz UART primitives.
* `rtl/weight_rom.sv` — synchronous `$readmemh` BRAM inference template.
* `rtl/uart_echo_top.sv` — first-board bring-up top level.
* `rtl/packet_parser.sv` — framed payload parser with XOR checksum validation.
* `rtl/packet_loop_top.sv` — parser, payload buffer, and UART response bridge.
* `rtl/attention_top.sv` — controller-to-systolic integration shell.
* `sim/tb_systolic_8x4.sv` — small RTL smoke test (Icarus/Verilator).
* `sim/tb_uart_echo.sv` — UART loopback smoke test with an accelerated divisor.
* `python/host.py` — optional `pyserial` transport client.
* `python/generate_artifacts.py` — generates softmax ROM and deterministic vectors.
* `python/model_export.py` — Q8.8 weight export and PC-side layer operations.
* `python/dataset.py` — deterministic shifted/noisy 16×16 pattern dataset.
* `python/train.py` — optional PyTorch export entry point.
* `python/app.py` — Tkinter drawing front end and UART inference client.
* `python/tiled_oracle.py` — 8×4 tiled matmul oracle.
* `python/measure_softmax.py` — fixed-point versus floating-point error report.
* `weights/README.md` — expected `$readmemh` layout.
* `docs/BUILD.md` — simulation and Gowin/open-source flow notes.

The array is deliberately exposed as a tile engine rather than hiding memory
policy inside it. A later controller can ping-pong BRAM buffers while one tile
computes. This keeps the DSP datapath independently testable.

The board-independent model/export path and the initial control-plane modules
are included. The exact BRAM address map and board pin constraints still need
to be reviewed against the target board revision before synthesis.

## Python quick start

```sh
python3 -m unittest discover -s python -p 'test_*.py'
```

The reference uses integer arithmetic throughout Q/K/V, score, and `S·V`.
Softmax is evaluated with a deterministic fixed-point exponential/reciprocal
implementation suitable as a software oracle; the generated FPGA LUT contents
must be compared against its output before integration.
