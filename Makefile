PYTHON ?= python3
IVERILOG ?= iverilog
VVP ?= vvp

.PHONY: test generate sim

test:
	PYTHONPATH=python $(PYTHON) -m unittest discover -s python -p 'test_*.py'

generate:
	$(PYTHON) python/generate_artifacts.py

dataset:
	$(PYTHON) python/dataset.py

sim:
	$(IVERILOG) -g2012 -o sim/tb_systolic sim/tb_systolic_8x4.sv rtl/systolic_8x4.sv
	$(VVP) sim/tb_systolic
	$(IVERILOG) -g2012 -o sim/tb_uart sim/tb_uart_echo.sv rtl/uart_echo_top.sv rtl/uart_rx.sv rtl/uart_tx.sv
	$(VVP) sim/tb_uart
