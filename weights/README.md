# Weights

Attention weights are no longer baked into the bitstream. The host loads them
over UART at run time (`TpuLink.load_weights`, or a `LOAD` frame per matrix),
so retraining needs no resynthesis. They live in BSRAM until power-off.

Each matrix is 64×64 signed Q8.8, sent as little-endian int16 in row-major
order W[k][c], the same orientation `attention_q8_8(x, wq, wk, wv, wo)` uses:
`x @ W`. In `model_export.py`, `export_layer` still writes the `$readmemh`-style
`layer{0,1}_{q,k,v,o}.mem` files (one four-hex-digit word per line, row-major),
and `host.read_matrix` reads them back for loading.

On the FPGA, word `layer*4096 + m*1024 + cg*64 + k` holds
`W_m[k][4cg .. 4cg+3]`, which is one B-edge column group of the systolic array.

`softmax_exp.mem` is the Q0.16 `exp(-n/16)` table (five hex digits, since
entry 0 is 65536) and is only a reference copy. The RTL uses the generated
`rtl/exp_rom.sv`.
