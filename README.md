# Tang Nano 20K FPGA-TPU

A 16-token, 64-wide single-head attention accelerator for the Sipeed Tang Nano
20K (Gowin GW2AR-LV18QN88C8/I7). The FPGA owns both layers' attention weights
and runs the whole attention block; the PC owns embedding, residual/layer
norm, feed-forward, and output projection.

One `ATTN` request computes, entirely on the FPGA and bit-exact with
`python/q8_attention.py`:

```
Q = X Wq   K = X Wk   V = X Wv          (Q8.8, each product floor-shifted by 8)
S = sat16((Q K^T) >> 3)                 (scaled dot product, d_k = 64)
P = softmax(S)                          (Q0.16, EXP_LUT + exact division, rows sum to 65536)
O = sat16((P V) >> 16) Wo
```

## Hardware

* **8×4 systolic array** of 18×18 MAC cells (32 of the 48 `MULT18X18` DSPs).
  Operands are 18 bits so Q0.16 probabilities up to 65536 share the datapath.
  A tile of K terms runs in K + 10 cycles; matrices are tiled 8 rows × 4 columns.
* **On-chip memories** (39 of 46 BSRAMs): 64 KiB of weights for two layers,
  a 512×144-bit activation memory in "8 rows per word" layout feeding the array's
  A edge, a 512×64-bit K^T/V memory feeding the B edge, and a 2 KiB UART buffer.
* **Softmax unit**: per-row max, `EXP_LUT` ROM, 33-cycle restoring divider and
  the remainder fix-up that makes each row sum to exactly 65536.
* **UART protocol** at 1 Mbaud (27 clocks per bit) over the on-board BL616 USB bridge, framed
  `A5 5A len16 payload xor`. Commands: ECHO, LOAD weights, ATTN, INFO.

At 27 MHz an attention layer computes in well under a millisecond; the
~2 KiB transfer each way dominates: about 51 ms per round trip at 1 Mbaud
(loading both layers' weights takes 0.7 s).

| LED (active low) | Meaning |
|---|---|
| 0 | heartbeat |
| 1 | request in progress |
| 2 | attention core computing |
| 3 | UART byte received |
| 4 | checksum error seen (sticky) |
| 5 | in reset (S1 held) |

## Layout

* `rtl/tang_nano_20k_top.sv` — board wrapper: power-on reset, S1 reset, LEDs.
* `rtl/tpu_top.sv` — command decoder, memories, request/response sequencing.
* `rtl/tpu_core.sv` — op sequencer, operand skew, tile writeback, memory map.
* `rtl/systolic_8x4.sv`, `rtl/pe.sv` — the MAC array.
* `rtl/softmax_unit.sv`, `rtl/exp_rom.sv` — softmax (ROM generated from the oracle).
* `rtl/packet_parser.sv`, `rtl/uart_rx.sv`, `rtl/uart_tx.sv`, `rtl/sdp_ram.sv`.
* `rtl/uart_echo_top.sv` — minimal bring-up top (byte echo).
* `constraints/tang_nano_20k.cst` — pins from Sipeed's official examples.
* `sim/` — array, UART and full end-to-end testbenches.
* `python/q8_attention.py` — bit-exact oracle and frame codec.
* `python/host.py` — `TpuLink` client and `selftest` CLI.
* `python/sim_vectors.py` — builds the end-to-end testbench script from the oracle.
* `python/model_export.py`, `train.py`, `dataset.py`, `app.py` — PC-side model and UI.

## Quick start

See [docs/BUILD.md](docs/BUILD.md) for the full flow. With OSS CAD Suite installed:

```sh
make test                         # Python oracle tests
make sim TOOLS=/path/to/oss-cad-suite
make bitstream TOOLS=/path/to/oss-cad-suite
make prog TOOLS=/path/to/oss-cad-suite      # SRAM; use 'flash' to persist
python python/host.py COM4 selftest         # random weights + inputs vs the oracle
```
