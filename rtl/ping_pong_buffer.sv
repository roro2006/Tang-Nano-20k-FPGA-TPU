module ping_pong_buffer #(
    parameter integer DEPTH = 1024,
    parameter integer WIDTH = 16
) (
    input logic clk, input logic rst, input logic select,
    input logic we, input logic [$clog2(DEPTH)-1:0] addr,
    input logic [WIDTH-1:0] din, output logic [WIDTH-1:0] dout
);
    logic [WIDTH-1:0] bank0 [0:DEPTH-1];
    logic [WIDTH-1:0] bank1 [0:DEPTH-1];
    always_ff @(posedge clk) begin
        if (we) begin
            if (select) bank1[addr] <= din;
            else bank0[addr] <= din;
        end
        dout <= select ? bank1[addr] : bank0[addr];
    end
endmodule
