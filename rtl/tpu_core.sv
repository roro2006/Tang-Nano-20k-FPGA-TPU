// One attention layer on the 8x4 systolic array, bit-exact with attention_q8_8.
//
// Memories (lane 0 in the low bits of every word):
//   A  512 x (8 x 18)  row-group words: word (g, k) holds M[8g + l][k] in lane l
//      X/O 0..127   Q 128..255   ATT 256..383   S 384..415   P 416..447
//   W  8192 x (4 x 16) weights: layer*4096 + m*1024 + cg*64 + k holds
//      W_m[k][4cg + c] in lane c, for m = q, k, v, o
//   C  512 x (4 x 16) column-group words, owned by this module:
//      K^T 0..255   (cg*64 + feature, lane c = token 4cg + c)
//      V   256..511 (256 + cg*16 + token, lane c = feature 4cg + c)
//
// Op sequence: Q = XWq, K = XWk, V = XWv, S = (Q K^T) >> 3, P = softmax(S),
// ATT = (P V) >> 16, O = ATT Wo. Each matmul runs 2 row groups x ncg column
// groups of 8x4 tiles, and every result is saturated to 16 bits.
module tpu_core #(
    parameter integer ACC_WIDTH = 40
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic layer,
    output logic busy,
    output logic done,
    output logic [8:0] a_raddr,
    input  logic [143:0] a_rdata,
    output logic a_we,
    output logic [8:0] a_waddr,
    output logic [143:0] a_wdata,
    output logic [12:0] w_raddr,
    input  logic [63:0] w_rdata
);
    localparam integer LW = 18;
    localparam logic [8:0] A_X = 9'd0, A_Q = 9'd128, A_ATT = 9'd256,
                           A_S = 9'd384, A_P = 9'd416;
    localparam logic [8:0] C_KT = 9'd0, C_V = 9'd256;
    localparam logic [2:0] OP_Q = 3'd0, OP_K = 3'd1, OP_V = 3'd2, OP_S = 3'd3,
                           OP_SOFTMAX = 3'd4, OP_ATT = 3'd5, OP_O = 3'd6;
    localparam logic [1:0] OUT_A = 2'd0, OUT_KT = 2'd1, OUT_V = 2'd2;

    typedef enum logic [2:0] {IDLE, OP_SETUP, TILE, WRITEBACK, SOFTMAX_RUN, FINISH} state_t;
    state_t state;
    logic [2:0] op;

    // Per-op configuration, decoded from op.
    logic [8:0] a_base, out_base;
    logic [6:0] kdim;
    logic out_k16;
    logic b_from_w;
    logic [12:0] b_base;
    logic b_stride16;
    logic [3:0] cg_last;
    logic [1:0] out_mode;
    logic [4:0] post_shift;
    logic shift8;

    always_comb begin
        a_base = A_X; kdim = 7'd64; b_from_w = 1'b1; b_stride16 = 1'b0;
        cg_last = 4'd15; out_mode = OUT_A; out_base = A_Q; out_k16 = 1'b0;
        post_shift = 5'd0; shift8 = 1'b1; b_base = {layer, 2'd0, 10'd0};
        case (op)
            OP_K: begin b_base = {layer, 2'd1, 10'd0}; out_mode = OUT_KT; end
            OP_V: begin b_base = {layer, 2'd2, 10'd0}; out_mode = OUT_V; end
            OP_S: begin
                a_base = A_Q; b_from_w = 1'b0; b_base = {4'd0, C_KT}; cg_last = 4'd3;
                out_base = A_S; out_k16 = 1'b1; post_shift = 5'd3;
            end
            OP_ATT: begin
                a_base = A_P; kdim = 7'd16; b_from_w = 1'b0; b_base = {4'd0, C_V};
                b_stride16 = 1'b1; out_base = A_ATT; post_shift = 5'd16; shift8 = 1'b0;
            end
            OP_O: begin a_base = A_ATT; b_base = {layer, 2'd3, 10'd0}; out_base = A_X; end
            default: ;
        endcase
    end

    // Tile iteration state.
    logic g;
    logic [3:0] cg;
    logic [6:0] n;
    logic [2:0] wb;
    logic rd_valid;

    logic [12:0] b_raddr;
    assign b_raddr = b_base + (b_stride16 ? {5'd0, cg, 4'd0} : {3'd0, cg, 6'd0}) + {6'd0, n};
    assign w_raddr = b_raddr;

    // C memory: K^T and V produced here and read back as B operands.
    logic c_we;
    logic [8:0] c_waddr;
    logic [63:0] c_wdata, c_rdata;
    sdp_ram #(.DEPTH(512), .WIDTH(64)) cmem(
        .clk(clk), .we(c_we), .waddr(c_waddr), .wdata(c_wdata),
        .raddr(b_raddr[8:0]), .rdata(c_rdata));

    // Operand skew: lane l of A is delayed l cycles, lane c of B c cycles.
    logic b_sel_w;
    logic [143:0] a_gated;
    logic [71:0] b_gated;
    logic [143:0] a_skew [0:6];
    logic [71:0] b_skew [0:2];
    logic [143:0] a_edge;
    logic [71:0] b_edge;

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : g_a
            assign a_gated[lane*LW +: LW] = rd_valid ? a_rdata[lane*LW +: LW] : '0;
            if (lane == 0) begin : g_a0
                assign a_edge[0 +: LW] = a_gated[0 +: LW];
            end else begin : g_an
                assign a_edge[lane*LW +: LW] = a_skew[lane-1][lane*LW +: LW];
            end
        end
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_b
            logic signed [15:0] b_word;
            assign b_word = b_sel_w ? w_rdata[lane*16 +: 16] : c_rdata[lane*16 +: 16];
            assign b_gated[lane*LW +: LW] = rd_valid ? LW'(b_word) : '0;
            if (lane == 0) begin : g_b0
                assign b_edge[0 +: LW] = b_gated[0 +: LW];
            end else begin : g_bn
                assign b_edge[lane*LW +: LW] = b_skew[lane-1][lane*LW +: LW];
            end
        end
    endgenerate

    logic array_clear, array_valid, array_done;
    logic [32*ACC_WIDTH-1:0] acc;
    systolic_8x4 #(.W(LW), .ACC_WIDTH(ACC_WIDTH)) array(
        .clk(clk), .rst(rst), .clear(array_clear), .valid_in(array_valid),
        .shift8(shift8), .klen(kdim), .a_edge(a_edge), .b_edge(b_edge),
        .done(array_done), .acc(acc));

    // Post-processed tile results: sat16(acc >>> post_shift).
    logic signed [15:0] result [0:7][0:3];
    genvar rr, cc;
    generate
        for (rr = 0; rr < 8; rr = rr + 1) begin : g_rr
            for (cc = 0; cc < 4; cc = cc + 1) begin : g_cc
                logic signed [ACC_WIDTH-1:0] raw, shifted;
                assign raw = acc[(rr*4 + cc)*ACC_WIDTH +: ACC_WIDTH];
                assign shifted = raw >>> post_shift;
                assign result[rr][cc] = (shifted > 32767) ? 16'sd32767 :
                                        (shifted < -32768) ? -16'sd32768 :
                                        shifted[15:0];
            end
        end
    endgenerate

    // Writeback word for the current wb index.
    logic [2:0] wb_last;
    logic [8:0] wb_addr;
    logic [143:0] wb_a_data;
    logic [63:0] wb_c_data;
    integer x;
    always_comb begin
        wb_last = (out_mode == OUT_A) ? 3'd3 : 3'd7;
        wb_a_data = '0;
        wb_c_data = '0;
        case (out_mode)
            OUT_KT: begin
                // Rows 4h..4h+3 of the tile are tokens 4(2g+h)..; column i is a feature.
                wb_addr = C_KT + {1'b0, g, wb[2], 6'd0} + {3'd0, cg, wb[1:0]};
                for (x = 0; x < 4; x = x + 1)
                    wb_c_data[x*16 +: 16] = result[{wb[2], 2'(x)}][wb[1:0]];
            end
            OUT_V: begin
                wb_addr = C_V + {1'b0, cg, 4'd0} + {5'd0, g, wb};
                for (x = 0; x < 4; x = x + 1)
                    wb_c_data[x*16 +: 16] = result[wb][x];
            end
            default: begin
                wb_addr = out_base + (g ? (out_k16 ? 9'd16 : 9'd64) : 9'd0)
                        + {3'd0, cg, wb[1:0]};
                for (x = 0; x < 8; x = x + 1)
                    wb_a_data[x*LW +: LW] = LW'(result[x][wb[1:0]]);
            end
        endcase
    end

    // Softmax shares the A memory ports while it runs.
    logic sm_start, sm_done, sm_we;
    logic [8:0] sm_raddr, sm_waddr;
    logic [143:0] sm_wdata;
    softmax_unit #(.LW(LW), .S_BASE(A_S), .P_BASE(A_P)) softmax(
        .clk(clk), .rst(rst), .start(sm_start), .done(sm_done),
        .raddr(sm_raddr), .rdata(a_rdata), .we(sm_we), .waddr(sm_waddr),
        .wdata(sm_wdata));

    logic tile_we;
    logic sm_active;
    assign sm_active = (state == SOFTMAX_RUN);
    assign a_raddr = sm_active ? sm_raddr : a_base + (g ? 9'(kdim) : 9'd0) + {2'd0, n};
    assign a_we    = sm_active ? sm_we : (tile_we && out_mode == OUT_A);
    assign a_waddr = sm_active ? sm_waddr : wb_addr;
    assign a_wdata = sm_active ? sm_wdata : wb_a_data;
    assign c_we    = tile_we && out_mode != OUT_A;
    assign c_waddr = wb_addr;
    assign c_wdata = wb_c_data;

    integer s;
    always_ff @(posedge clk) begin
        done <= 1'b0;
        sm_start <= 1'b0;
        array_clear <= 1'b0;
        array_valid <= 1'b0;
        tile_we <= 1'b0;
        // Read data for address n appears one cycle later.
        rd_valid <= (state == TILE) && (n < kdim);
        b_sel_w <= b_from_w;
        a_skew[0] <= a_gated;
        b_skew[0] <= b_gated;
        for (s = 1; s < 7; s = s + 1) a_skew[s] <= a_skew[s-1];
        for (s = 1; s < 3; s = s + 1) b_skew[s] <= b_skew[s-1];

        if (rst) begin
            state <= IDLE;
            busy <= 1'b0;
            op <= OP_Q;
            g <= 1'b0;
            cg <= '0;
            n <= '0;
            wb <= '0;
            rd_valid <= 1'b0;
        end else case (state)
            IDLE: if (start) begin
                busy <= 1'b1;
                op <= OP_Q;
                state <= OP_SETUP;
            end
            OP_SETUP: begin
                g <= 1'b0;
                cg <= '0;
                n <= '0;
                if (op == OP_SOFTMAX) begin
                    sm_start <= 1'b1;
                    state <= SOFTMAX_RUN;
                end else begin
                    array_clear <= 1'b1;
                    state <= TILE;
                end
            end
            TILE: begin
                // Cycle n = 0 issues the first read while the array clears;
                // cycles n = 1 .. kdim + 10 carry the skewed edges.
                array_valid <= (n <= kdim + 7'd9);
                n <= n + 1'b1;
                if (array_done) begin
                    wb <= '0;
                    state <= WRITEBACK;
                end
            end
            WRITEBACK: begin
                tile_we <= 1'b1;
                if (tile_we) begin
                    // The word for wb is written on this edge.
                    tile_we <= wb != wb_last;
                    if (wb != wb_last) begin
                        wb <= wb + 1'b1;
                    end else begin
                        n <= '0;
                        if (cg != cg_last) begin
                            cg <= cg + 1'b1;
                            array_clear <= 1'b1;
                            state <= TILE;
                        end else if (!g) begin
                            cg <= '0;
                            g <= 1'b1;
                            array_clear <= 1'b1;
                            state <= TILE;
                        end else begin
                            op <= op + 1'b1;
                            state <= state_t'((op == OP_O) ? FINISH : OP_SETUP);
                        end
                    end
                end
            end
            SOFTMAX_RUN: if (sm_done) begin
                op <= op + 1'b1;
                state <= OP_SETUP;
            end
            FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end
endmodule
