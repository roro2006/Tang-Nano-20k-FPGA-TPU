module weight_rom #(
    parameter integer DEPTH = 32768,
    parameter MEMFILE = "weights/layer0_q.mem"
) (
    input logic clk,
    input logic [$clog2(DEPTH)-1:0] addr,
    output logic signed [15:0] data
);
    logic signed [15:0] mem [0:DEPTH-1];
    initial $readmemh(MEMFILE, mem);
    always_ff @(posedge clk)
        data <= mem[addr];
endmodule

