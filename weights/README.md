# Weight ROM format

Export signed Q8.8 values as four-hex-digit two's-complement words, one value
per line, in row-major order. Keep separate files for `layer{0,1}_{q,k,v,o}.mem`
or concatenate them with a documented base address. `$readmemh` loads these
files into BRAM inference.

`softmax_exp.mem` stores Q0.16 `exp(-n/16)` values. `softmax_recip.mem`
stores Q8.16 reciprocal values indexed by the normalized 16-bit denominator;
zero maps to zero. The softmax datapath saturates outputs and assigns any
truncation remainder to the row maximum so every row sums to 65536.

The four 64×64 matrices per layer consume 32 KiB per layer, or 64 KiB for two
layers. Embedding,
feed-forward, and output weights remain on the PC.
