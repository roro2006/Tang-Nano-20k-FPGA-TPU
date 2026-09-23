module attention_controller (
    input logic clk, input logic rst, input logic start,
    output logic busy, output logic done,
    output logic tile_start, output logic tile_valid
);
    typedef enum logic [2:0] {IDLE, Q_PROJ, K_PROJ, V_PROJ, SCORES, SOFTMAX, SV, OUTPUT} state_t;
    state_t state;
    logic [7:0] tile;
    always_ff @(posedge clk) begin
        tile_start <= 0; tile_valid <= 0; done <= 0;
        if (rst) begin state <= IDLE; busy <= 0; tile <= 0; end
        else case (state)
            IDLE: if (start) begin state <= Q_PROJ; busy <= 1; tile <= 0; end
            Q_PROJ, K_PROJ, V_PROJ, SV, OUTPUT: begin
                tile_start <= tile == 0;
                tile_valid <= 1;
                if (tile == 31) begin
                    tile <= 0;
                    state <= state == Q_PROJ ? K_PROJ :
                             state == K_PROJ ? V_PROJ :
                             state == V_PROJ ? SCORES :
                             state == SV ? OUTPUT : IDLE;
                    if (state == OUTPUT) begin busy <= 0; done <= 1; end
                end else tile <= tile + 1;
            end
            SCORES: state <= SOFTMAX;
            SOFTMAX: state <= SV;
            default: state <= IDLE;
        endcase
    end
endmodule
