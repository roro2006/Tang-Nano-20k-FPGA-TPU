module systolic_8x4 #(
    parameter ACC_WIDTH = 40
) (
    input logic clk,
    input logic rst,
    input logic start,
    input logic valid_in,
    input logic signed [15:0] a_in [0:7],
    input logic signed [15:0] b_in [0:3],
    output logic done,
    output logic signed [ACC_WIDTH-1:0] result [0:7][0:3]
);
    logic signed [ACC_WIDTH-1:0] accum [0:7][0:3];
    logic signed [15:0] a_pipe [0:7][0:3];
    logic signed [15:0] b_pipe [0:7][0:3];
    logic [6:0] count;
    integer r, c;

    // A flows left-to-right and B flows top-to-bottom. Edge values must be
    // staggered by row/column respectively: A[r][k] is presented at k+r and
    // B[k][c] at k+c. This gives every cell the matching pair at k+r+c.
    always_ff @(posedge clk) begin
        if (rst) begin
            count <= '0;
            done <= 1'b0;
            for (r = 0; r < 8; r = r + 1)
                for (c = 0; c < 4; c = c + 1) begin
                    accum[r][c] <= '0;
                    a_pipe[r][c] <= '0;
                    b_pipe[r][c] <= '0;
                end
        end else begin
            done <= 1'b0;
            if (start) begin
                count <= 0;
                for (r = 0; r < 8; r = r + 1)
                    for (c = 0; c < 4; c = c + 1) begin
                        accum[r][c] <= '0;
                        a_pipe[r][c] <= '0;
                        b_pipe[r][c] <= '0;
                    end
            end else if (valid_in) begin
                for (r = 0; r < 8; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        accum[r][c] <= accum[r][c] + a_pipe[r][c] * b_pipe[r][c];
                        if (c == 0) a_pipe[r][c] <= a_in[r];
                        else a_pipe[r][c] <= a_pipe[r][c-1];
                        if (r == 0) b_pipe[r][c] <= b_in[c];
                        else b_pipe[r][c] <= b_pipe[r-1][c];
                    end
                end
                if (count == 7'd73) begin
                    done <= 1'b1;
                    for (r = 0; r < 8; r = r + 1)
                        for (c = 0; c < 4; c = c + 1)
                            result[r][c] <= accum[r][c] +
                                a_pipe[r][c] * b_pipe[r][c];
                end
                count <= count + 1'b1;
            end
        end
    end
endmodule
