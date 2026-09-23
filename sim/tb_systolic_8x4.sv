module tb_systolic_8x4;
    logic clk = 0, rst = 1, start = 0, valid_in = 0, done;
    logic signed [15:0] a [0:7], b [0:3];
    logic signed [39:0] result [0:7][0:3];
    integer i, j, k;
    systolic_8x4 dut(.*);
    always #5 clk = ~clk;
    initial begin
        for (i = 0; i < 8; i = i + 1) a[i] = i + 1;
        for (j = 0; j < 4; j = j + 1) b[j] = (j + 1) * 2;
        #12 rst = 0; start = 1; #10 start = 0;
        for (k = 0; k < 74; k = k + 1) begin
            for (i = 0; i < 8; i = i + 1)
                a[i] = (k >= i && k < i + 64) ? i + 1 : 0;
            for (j = 0; j < 4; j = j + 1)
                b[j] = (k >= j && k < j + 64) ? (j + 1) * 2 : 0;
            @(negedge clk); valid_in = 1; @(negedge clk); valid_in = 0;
        end
        @(negedge clk);
        if (!done || result[3][2] != (4 * 6 * 64))
            $fatal(1, "systolic result mismatch");
        $display("systolic_8x4 PASS");
        $finish;
    end
endmodule
