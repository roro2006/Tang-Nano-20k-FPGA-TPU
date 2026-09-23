module softmax_lut #(
    parameter EXP_FILE = "weights/softmax_exp.mem"
) (
    input logic clk, input logic start,
    input logic signed [15:0] score [0:15],
    output logic busy, output logic done,
    output logic [15:0] probability [0:15]
);
    logic [16:0] exp_rom [0:255];
    logic [16:0] exps [0:15];
    logic [23:0] total;
    logic [3:0] index;
    logic signed [15:0] maximum;
    integer i;
    initial $readmemh(EXP_FILE, exp_rom);
    always_ff @(posedge clk) begin
        done <= 0;
        if (start && !busy) begin
            maximum <= score[0];
            for (i = 1; i < 16; i = i + 1)
                if (score[i] > maximum) maximum <= score[i];
            total <= 0;
            index <= 0;
            busy <= 1;
        end else if (busy) begin
            if (index < 16) begin
                exps[index] <= exp_rom[((maximum - score[index]) >>> 4) > 255 ?
                    255 : ((maximum - score[index]) >>> 4)];
                total <= total + exps[index];
                index <= index + 1;
            end else begin
                for (i = 0; i < 16; i = i + 1)
                    probability[i] <= total ? ((exps[i] << 16) / total) : 0;
                busy <= 0;
                done <= 1;
            end
        end
    end
endmodule
