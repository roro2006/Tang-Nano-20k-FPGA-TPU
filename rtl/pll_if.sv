module pll_if (
    input logic clk27, input logic rst, output logic clk81, output logic locked
);
    // Replace with the Gowin PLL primitive in the board build.
    assign clk81 = clk27;
    assign locked = !rst;
endmodule
