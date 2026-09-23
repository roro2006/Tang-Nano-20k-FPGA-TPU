# Build and validation

## Python oracle

From the project root:

```sh
PYTHONPATH=python python3 -m unittest discover -s python -p 'test_*.py'
```

Generate the LUT and a 16×64 identity-weight vector:

```sh
make generate
```

Create the deterministic starter dataset:

```sh
make dataset
```

## RTL simulation

With Icarus Verilog:

```sh
iverilog -g2012 -o sim/tb sim/tb_systolic_8x4.sv rtl/systolic_8x4.sv
vvp sim/tb
```

Run the UART smoke test with a small simulated divisor:

```sh
iverilog -g2012 -o sim/tb_uart sim/tb_uart_echo.sv rtl/uart_echo_top.sv \
  rtl/uart_rx.sv rtl/uart_tx.sv
vvp sim/tb_uart
```

The current UART divisor is `27_000_000 / 115200 ≈ 234`. Validate echo first,
then characterize the BL616 link before selecting a faster divisor.

## Hardware flow

For Gowin EDA, create a Tang Nano 20K project, add the RTL and a board-specific
PLL wrapper, constrain the 27 MHz input and LED/UART pins, and synthesize.
For the open flow, use Yosys plus apicula/nextpnr-gowin and program with
`openFPGALoader`. Keep the Linux JTAG udev rule disabled while testing UART if
it interferes with the BL616 USB bridge.

Start at 81 MHz or below if timing at 100 MHz fails. The tile engine accepts staggered edge streams for 64 K terms and asserts
`done` after the 74-cycle fill/compute/drain interval. The eventual controller
should hide BRAM loads with ping-pong buffers.

The exact pin constraints are intentionally left as placeholders in
`constraints/tang_nano_20k.cst`; obtain them from the board revision schematic.
