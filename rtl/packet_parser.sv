module packet_parser #(
    parameter integer MAX_PAYLOAD = 2048
) (
    input  logic clk,
    input  logic rst,
    input  logic [7:0] byte_in,
    input  logic byte_valid,
    output logic [7:0] payload_data,
    output logic payload_valid,
    output logic packet_valid,
    output logic packet_error,
    output logic [$clog2(MAX_PAYLOAD+1)-1:0] payload_length
);
    typedef enum logic [2:0] {WAIT_A5, WAIT_5A, LEN_LO, LEN_HI, PAYLOAD, CHECKSUM} state_t;
    state_t state;
    logic [15:0] expected_length, index;
    logic [7:0] checksum;

    always_ff @(posedge clk) begin
        payload_valid <= 1'b0;
        packet_valid <= 1'b0;
        packet_error <= 1'b0;
        if (rst) begin
            state <= WAIT_A5;
            expected_length <= '0;
            index <= '0;
            checksum <= '0;
            payload_length <= '0;
        end else if (byte_valid) begin
            case (state)
                WAIT_A5: state <= (byte_in == 8'hA5) ? WAIT_5A : WAIT_A5;
                WAIT_5A: state <= (byte_in == 8'h5A) ? LEN_LO : WAIT_A5;
                LEN_LO: begin expected_length[7:0] <= byte_in; state <= LEN_HI; end
                LEN_HI: begin
                    expected_length[15:8] <= byte_in;
                    index <= 0;
                    checksum <= 0;
                    state <= ({byte_in, expected_length[7:0]} > MAX_PAYLOAD) ? WAIT_A5 :
                             ({byte_in, expected_length[7:0]} == 0) ? CHECKSUM : PAYLOAD;
                end
                PAYLOAD: begin
                    if (index < MAX_PAYLOAD) begin
                        payload_data <= byte_in;
                        payload_valid <= 1'b1;
                        checksum <= checksum ^ byte_in;
                    end
                    index <= index + 1'b1;
                    if (index + 1 >= expected_length)
                        state <= CHECKSUM;
                end
                CHECKSUM: begin
                    if (byte_in == checksum) begin
                        payload_length <= expected_length[$bits(payload_length)-1:0];
                        packet_valid <= 1'b1;
                    end else packet_error <= 1'b1;
                    state <= WAIT_A5;
                end
                default: state <= WAIT_A5;
            endcase
        end
    end
endmodule
