module pe #(
    parameter ACC_WIDTH = 40
) (
    input  logic clk,
    input  logic rst,
    input  logic clear,
    input  logic valid_in,
    input  logic signed [15:0] a_in,
    input  logic signed [15:0] b_in,
    output logic valid_out,
    output logic signed [15:0] a_out,
    output logic signed [15:0] b_out,
    output logic signed [ACC_WIDTH-1:0] acc
);
    always_ff @(posedge clk) begin
        if (rst) begin
            a_out <= '0;
            b_out <= '0;
            acc <= '0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            a_out <= a_in;
            b_out <= b_in;
            if (clear)
                acc <= '0;
            else if (valid_in)
                acc <= acc + (a_in * b_in);
        end
    end
endmodule

