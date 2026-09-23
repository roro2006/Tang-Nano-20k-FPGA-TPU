module packet_loop_top #(
    parameter integer CLKS_PER_BIT = 234,
    parameter integer MAX_PAYLOAD = 2048
) (
    input logic clk, input logic rst, input logic uart_rx_pin,
    output logic uart_tx_pin, output logic packet_error
);
    logic [7:0] rx_byte, payload_byte;
    logic rx_valid, payload_valid, packet_valid, tx_start, tx_busy;
    logic [$clog2(MAX_PAYLOAD+1)-1:0] payload_length;
    logic [7:0] memory [0:MAX_PAYLOAD-1];
    logic [$clog2(MAX_PAYLOAD+1)-1:0] write_index, read_index;
    logic [2:0] state;
    logic [7:0] tx_data;
    logic [7:0] response_checksum;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx(
        .clk(clk), .rst(rst), .rx(uart_rx_pin), .data(rx_byte), .valid(rx_valid));
    packet_parser #(.MAX_PAYLOAD(MAX_PAYLOAD)) parser(
        .clk(clk), .rst(rst), .byte_in(rx_byte), .byte_valid(rx_valid),
        .payload_data(payload_byte), .payload_valid(payload_valid),
        .packet_valid(packet_valid), .packet_error(packet_error),
        .payload_length(payload_length));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx(
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data),
        .tx(uart_tx_pin), .busy(tx_busy));

    always_ff @(posedge clk) begin
        tx_start <= 0;
        if (rst) begin
            write_index <= 0; read_index <= 0; state <= 0; tx_data <= 0;
            response_checksum <= 0;
        end else begin
            if (payload_valid && write_index < MAX_PAYLOAD) begin
                memory[write_index] <= payload_byte;
                write_index <= write_index + 1;
                response_checksum <= response_checksum ^ payload_byte;
            end
            if (packet_valid) begin
                read_index <= 0;
                write_index <= 0;
                state <= 1;
            end else if (!tx_busy) case (state)
                1: begin tx_data <= 8'hA5; tx_start <= 1; state <= 2; end
                2: begin tx_data <= 8'h5A; tx_start <= 1; state <= 3; end
                3: begin tx_data <= payload_length[7:0]; tx_start <= 1; state <= 4; end
                4: begin tx_data <= payload_length[15:8]; tx_start <= 1;
                    state <= payload_length == 0 ? 6 : 5; end
                5: begin
                    tx_data <= memory[read_index]; tx_start <= 1;
                    read_index <= read_index + 1;
                    if (read_index + 1 >= payload_length) state <= 6;
                end
                6: begin
                    tx_data <= response_checksum; tx_start <= 1;
                    response_checksum <= 0; state <= 0;
                end
                default: state <= 0;
            endcase
        end
    end
endmodule
