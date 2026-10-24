// Board-independent accelerator top: UART framing, commands and memories.
//
// Request payload:  op, arg, data...      Response payload: op, status, data...
//   01 ECHO  arg, up to 2048 bytes       -> 01 00 + the same bytes
//   02 LOAD  sel = layer*4 + m (m = q,k,v,o), 64x64 int16 LE row-major W[k][c]
//                                         -> 02 00 (01 on bad length/sel)
//   03 ATTN  layer, 16x64 int16 LE row-major X
//                                         -> 03 00 + 16x64 int16 LE attention
//   04 INFO                               -> 04 00 'T' 'P' 'U' version
//   bad checksum                          -> EE 01
// LOAD writes weights as bytes arrive, so a LOAD answered with EE must be
// resent. The host waits for each response before sending the next request.
module tpu_top #(
    parameter integer CLKS_PER_BIT = 234,
    parameter integer TIMEOUT_CLKS = 2_700_000
) (
    input  logic clk,
    input  logic rst,
    input  logic uart_rx_pin,
    output logic uart_tx_pin,
    output logic busy,
    output logic computing,
    output logic rx_activity,
    output logic error_seen
);
    localparam integer MAX_PAYLOAD = 8194;
    localparam logic [7:0] OP_ECHO = 8'h01, OP_LOAD = 8'h02, OP_ATTN = 8'h03,
                           OP_INFO = 8'h04, OP_ERROR = 8'hEE;
    localparam logic [7:0] VERSION = 8'h01;
    localparam logic [1:0] SRC_NONE = 2'd0, SRC_RX = 2'd1, SRC_O = 2'd2, SRC_INFO = 2'd3;

    // UART and framing.
    logic [7:0] rx_byte, payload_byte;
    logic rx_valid, payload_valid, packet_start, packet_valid, packet_error;
    logic [13:0] payload_length;
    logic tx_start, tx_busy;
    logic [7:0] tx_data;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx(
        .clk(clk), .rst(rst), .rx(uart_rx_pin), .data(rx_byte), .valid(rx_valid));
    packet_parser #(.MAX_PAYLOAD(MAX_PAYLOAD), .TIMEOUT_CLKS(TIMEOUT_CLKS)) parser(
        .clk(clk), .rst(rst), .byte_in(rx_byte), .byte_valid(rx_valid),
        .payload_data(payload_byte), .payload_valid(payload_valid),
        .packet_start(packet_start), .packet_valid(packet_valid),
        .packet_error(packet_error), .payload_length(payload_length));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx(
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data),
        .tx(uart_tx_pin), .busy(tx_busy));

    // Memories.
    logic a_we;
    logic [8:0] a_waddr, a_raddr;
    logic [143:0] a_wdata, a_rdata;
    sdp_ram #(.DEPTH(512), .WIDTH(144)) amem(
        .clk(clk), .we(a_we), .waddr(a_waddr), .wdata(a_wdata),
        .raddr(a_raddr), .rdata(a_rdata));

    logic w_we;
    logic [12:0] w_waddr, w_raddr;
    logic [63:0] w_wdata, w_rdata;
    sdp_ram #(.DEPTH(8192), .WIDTH(64)) wmem(
        .clk(clk), .we(w_we), .waddr(w_waddr), .wdata(w_wdata),
        .raddr(w_raddr), .rdata(w_rdata));

    logic rx_we;
    logic [10:0] rx_waddr, rx_raddr;
    logic [7:0] rx_wdata, rx_rdata;
    sdp_ram #(.DEPTH(2048), .WIDTH(8)) rxbuf(
        .clk(clk), .we(rx_we), .waddr(rx_waddr), .wdata(rx_wdata),
        .raddr(rx_raddr), .rdata(rx_rdata));

    // Attention core.
    logic core_start, core_busy, core_done, layer;
    logic core_a_we;
    logic [8:0] core_a_raddr, core_a_waddr;
    logic [143:0] core_a_wdata;
    tpu_core core(
        .clk(clk), .rst(rst), .start(core_start), .layer(layer),
        .busy(core_busy), .done(core_done),
        .a_raddr(core_a_raddr), .a_rdata(a_rdata), .a_we(core_a_we),
        .a_waddr(core_a_waddr), .a_wdata(core_a_wdata),
        .w_raddr(w_raddr), .w_rdata(w_rdata));

    typedef enum logic [1:0] {T_IDLE, T_REPACK, T_RUN, T_RESP} top_state_t;
    typedef enum logic [1:0] {R_FETCH, R_SEND, R_WAIT} resp_state_t;
    top_state_t state;
    resp_state_t rstate;

    // Request capture.
    logic [13:0] byte_idx;
    logic [12:0] data_idx;
    logic [7:0] opcode, arg;
    logic [55:0] w_word;
    assign data_idx = 13'(byte_idx - 14'd2);

    // Response sequencing: pos 0..3 header, 4 op, 5 status, data, checksum.
    logic [7:0] resp_op, resp_status;
    logic [1:0] resp_src;
    logic [11:0] resp_data_len;
    logic [13:0] pos;
    logic [11:0] resp_i;
    logic [13:0] resp_payload_len;
    logic [7:0] resp_checksum;
    logic [7:0] resp_byte;
    assign resp_payload_len = {2'd0, resp_data_len} + 14'd2;
    assign resp_i = 12'(pos - 14'd6);

    always_comb begin
        if (pos == 0) resp_byte = 8'hA5;
        else if (pos == 1) resp_byte = 8'h5A;
        else if (pos == 2) resp_byte = resp_payload_len[7:0];
        else if (pos == 3) resp_byte = {2'd0, resp_payload_len[13:8]};
        else if (pos == 4) resp_byte = resp_op;
        else if (pos == 5) resp_byte = resp_status;
        else if (pos < 14'd6 + {2'd0, resp_data_len}) begin
            case (resp_src)
                SRC_RX: resp_byte = rx_rdata;
                // Byte i of row-major O: row i[10:7] (lane i[9:7]), column i[6:1].
                SRC_O: resp_byte = resp_i[0] ? a_rdata[resp_i[9:7]*18 + 8 +: 8]
                                             : a_rdata[resp_i[9:7]*18 +: 8];
                SRC_INFO: case (resp_i[1:0])
                    2'd0: resp_byte = 8'h54;
                    2'd1: resp_byte = 8'h50;
                    2'd2: resp_byte = 8'h55;
                    default: resp_byte = VERSION;
                endcase
                default: resp_byte = 8'h00;
            endcase
        end else resp_byte = resp_checksum;
    end

    // Repack: rx bytes in row-major order -> A words of 8 rows at column k.
    // rp = {g, k[5:0], l[2:0], hi}; the byte address is {g, l, k, hi}.
    logic [10:0] rp, rp_d;
    logic rp_active, rp_valid;
    logic [143:0] rp_word, rp_next;
    always_comb begin
        rp_next = rp_word;
        rp_next[rp_d[3:1]*18 + rp_d[0]*8 +: 8] = rx_rdata;
        if (rp_d[0])
            rp_next[rp_d[3:1]*18 + 16 +: 2] = {2{rx_rdata[7]}};
    end

    // Memory port ownership.
    always_comb begin
        if (state == T_REPACK)
            rx_raddr = {rp[10], rp[3:1], rp[9:4], rp[0]};
        else
            rx_raddr = resp_i[10:0];
        if (state == T_RUN) begin
            a_raddr = core_a_raddr;
            a_we = core_a_we;
            a_waddr = core_a_waddr;
            a_wdata = core_a_wdata;
        end else begin
            // O row r lives in word (r[3], c) of the X region.
            a_raddr = {2'd0, resp_i[10], resp_i[6:1]};
            a_we = rp_valid && rp_d[3:0] == 4'hF;
            a_waddr = {2'd0, rp_d[10], rp_d[9:4]};
            a_wdata = rp_next;
        end
    end

    assign busy = state != T_IDLE;
    assign computing = core_busy;

    always_ff @(posedge clk) begin
        w_we <= 1'b0;
        rx_we <= 1'b0;
        tx_start <= 1'b0;
        core_start <= 1'b0;
        rx_activity <= rx_valid;
        if (rst) begin
            state <= T_IDLE;
            rstate <= R_FETCH;
            byte_idx <= '0;
            opcode <= '0;
            arg <= '0;
            pos <= '0;
            rp <= '0;
            rp_active <= 1'b0;
            rp_valid <= 1'b0;
            layer <= 1'b0;
            error_seen <= 1'b0;
        end else begin
            if (packet_start)
                byte_idx <= '0;
            if (payload_valid && state == T_IDLE) begin
                byte_idx <= byte_idx + 1'b1;
                if (byte_idx == 0) opcode <= payload_byte;
                if (byte_idx == 1) arg <= payload_byte;
                if (byte_idx >= 2) begin
                    if (opcode == OP_LOAD) begin
                        if (data_idx[2:0] != 3'd7) begin
                            w_word[data_idx[2:0]*8 +: 8] <= payload_byte;
                        end else begin
                            // Word data_idx/8 = k*16 + cg holds W[k][4cg..4cg+3].
                            w_we <= 1'b1;
                            w_waddr <= {arg[2:0], data_idx[6:3], data_idx[12:7]};
                            w_wdata <= {payload_byte, w_word};
                        end
                    end else if (data_idx < 13'd2048) begin
                        rx_we <= 1'b1;
                        rx_waddr <= data_idx[10:0];
                        rx_wdata <= payload_byte;
                    end
                end
            end

            case (state)
                T_IDLE: begin
                    pos <= '0;
                    resp_checksum <= '0;
                    rstate <= R_FETCH;
                    resp_src <= SRC_NONE;
                    resp_data_len <= '0;
                    resp_status <= 8'h00;
                    resp_op <= opcode;
                    if (packet_error) begin
                        error_seen <= 1'b1;
                        resp_op <= OP_ERROR;
                        resp_status <= 8'h01;
                        state <= T_RESP;
                    end else if (packet_valid && payload_length == 0) begin
                        resp_op <= OP_ERROR;
                        resp_status <= 8'h02;
                        state <= T_RESP;
                    end else if (packet_valid) begin
                        state <= T_RESP;
                        case (opcode)
                            OP_ECHO: if (payload_length <= 14'd2050) begin
                                resp_src <= SRC_RX;
                                resp_data_len <= (payload_length > 14'd2) ?
                                                 12'(payload_length - 14'd2) : 12'd0;
                            end else resp_status <= 8'h01;
                            OP_LOAD: if (payload_length != 14'd8194 || arg > 8'd7)
                                resp_status <= 8'h01;
                            OP_ATTN: if (payload_length == 14'd2050 && arg < 8'd2) begin
                                layer <= arg[0];
                                rp <= '0;
                                rp_active <= 1'b1;
                                state <= T_REPACK;
                            end else resp_status <= 8'h01;
                            OP_INFO: begin
                                resp_src <= SRC_INFO;
                                resp_data_len <= 12'd4;
                            end
                            default: resp_status <= 8'h02;
                        endcase
                    end
                end
                T_REPACK: begin
                    rp_valid <= rp_active;
                    rp_d <= rp;
                    if (rp_valid) rp_word <= rp_next;
                    if (rp_active) begin
                        rp <= rp + 1'b1;
                        if (rp == 11'h7FF) rp_active <= 1'b0;
                    end else if (!rp_valid) begin
                        core_start <= 1'b1;
                        state <= T_RUN;
                    end
                end
                T_RUN: if (core_done) begin
                    resp_src <= SRC_O;
                    resp_data_len <= 12'd2048;
                    state <= T_RESP;
                end
                T_RESP: case (rstate)
                    // Addresses derive from pos; give the BSRAM one cycle.
                    R_FETCH: rstate <= R_SEND;
                    R_SEND: if (!tx_busy) begin
                        tx_data <= resp_byte;
                        tx_start <= 1'b1;
                        if (pos >= 14'd4) resp_checksum <= resp_checksum ^ resp_byte;
                        rstate <= R_WAIT;
                    end
                    R_WAIT: if (tx_busy) begin
                        if (pos == 14'd6 + {2'd0, resp_data_len}) begin
                            state <= T_IDLE;
                        end else begin
                            pos <= pos + 1'b1;
                            rstate <= R_FETCH;
                        end
                    end
                    default: rstate <= R_FETCH;
                endcase
                default: state <= T_IDLE;
            endcase
        end
    end
endmodule
