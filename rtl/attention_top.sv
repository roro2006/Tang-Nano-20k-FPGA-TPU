module attention_top (
    input logic clk, input logic rst, input logic start,
    output logic busy, output logic done
);
    logic tile_start, tile_valid;
    logic signed [15:0] a [0:7], b [0:3];
    logic signed [39:0] tile_result [0:7][0:3];
    logic tile_done;
    integer i;

    attention_controller controller(
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .tile_start(tile_start), .tile_valid(tile_valid));
    systolic_8x4 array(
        .clk(clk), .rst(rst), .start(tile_start), .valid_in(tile_valid),
        .a_in(a), .b_in(b), .done(tile_done), .result(tile_result));

    // BRAM schedulers drive a/b in the production build. Keeping the edge
    // ports explicit makes this shell suitable for synthesis and integration.
    always_comb begin
        for (i = 0; i < 8; i = i + 1) a[i] = 0;
        for (i = 0; i < 4; i = i + 1) b[i] = 0;
    end
endmodule
