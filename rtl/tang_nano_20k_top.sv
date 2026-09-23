module tang_nano_20k_top (
    input logic clk27, input logic rst, input logic uart_rx_pin,
    output logic uart_tx_pin, output logic led
);
    logic clk, locked;
    pll_if pll(.clk27(clk27), .rst(rst), .clk81(clk), .locked(locked));
    uart_echo_top #(.CLKS_PER_BIT(234)) echo(
        .clk27(clk), .rst(rst || !locked), .uart_rx_pin(uart_rx_pin),
        .uart_tx_pin(uart_tx_pin)
    );
    assign led = locked;
endmodule
