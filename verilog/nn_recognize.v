`timescale 1ns / 1ps

module nn_recognize();
    
    // for reading input predict image
    parameter IMG_SIZE = 784,
    IMG_FILE = "./data/hex_img.hex";
    reg [7:0] img_memory [0:IMG_SIZE - 1];
    real img_data [0:IMG_SIZE - 1];
    
    // for reading layer1_weight
    parameter W1_SIZE = 100352,
    W1_FILE = "./data/layer1_weight.hex";
    reg [31:0] w1_memory [0:W1_SIZE - 1];
    real w1_data [0:W1_SIZE - 1];
    
    // for reading layer1_bias
    parameter B1_SIZE = 128,
    B1_FILE = "./data/layer1_bias.hex";
    reg [31:0] b1_memory [0:B1_SIZE - 1];
    real b1_data [0:B1_SIZE - 1];
    
    // for reading layer2_weight
    parameter W2_SIZE = 1280,
    W2_FILE = "./data/layer2_weight.hex";
    reg [31:0] w2_memory [0:W2_SIZE - 1];
    real w2_data [0:W2_SIZE - 1];
    
    // for reading layer2_bias
    parameter B2_SIZE = 10,
    B2_FILE = "./data/layer2_bias.hex";
    reg [31:0] b2_memory [0:B2_SIZE - 1];
    real b2_data [0:B2_SIZE - 1];
    
    // for neural network calculation
    integer i, j, k;
    integer row1, col1, col2;
    real sum;
    real x1 [0:127];
    real x2 [0:9];
    real max_value;
    integer predicted_digit;
    
    // signals
    reg CLK;    
    reg [2:0] next_state;
    reg [2:0] state;
    reg finish_read;
    reg start_nn_cal;
    
    initial begin
        CLK = 0;
        next_state = 0;
        state = 0;
        finish_read = 0;
        start_nn_cal = 0;
        // Read image
        $readmemh(IMG_FILE, img_memory, 0, IMG_SIZE - 1);
        for(i = 0; i < IMG_SIZE; i = i + 1) begin
            img_data[i] = img_memory[i];
            img_data[i] = img_data[i] / 255;
            //$display("%d", img_memory[i]);
        end
        // Read layer1 weight
        $readmemh(W1_FILE, w1_memory, 0, W1_SIZE - 1);
        for(i = 0; i < W1_SIZE; i = i + 1) begin
            if(w1_memory[i] >= 134217728) begin
                w1_data[i] = w1_memory[i] - 134217728;
                w1_data[i] = w1_data[i] / 100000000 * (-1);
            end
            else begin
                w1_data[i] = w1_memory[i];
                w1_data[i] = w1_data[i] / 100000000;
            end
            //$display("%0.8f", w1_data[i]);
        end
        // Read layer1 bias
        $readmemh(B1_FILE, b1_memory, 0, B1_SIZE - 1);
        for(i = 0; i < B1_SIZE; i = i + 1) begin
            if(b1_memory[i] >= 134217728) begin
                b1_data[i] = b1_memory[i] - 134217728;
                b1_data[i] = b1_data[i] / 100000000 * (-1);
            end
            else begin
                b1_data[i] = b1_memory[i];
                b1_data[i] = b1_data[i] / 100000000;
            end
            //$display("%0.8f", b1_data[i]);
        end
        // Read layer2 weight
        $readmemh(W2_FILE, w2_memory, 0, W2_SIZE - 1);
        for(i = 0; i < W2_SIZE; i = i + 1) begin
            if(w2_memory[i] >= 134217728) begin
                w2_data[i] = w2_memory[i] - 134217728;
                w2_data[i] = w2_data[i] / 100000000 * (-1);
            end
            else begin
                w2_data[i] = w2_memory[i];
                w2_data[i] = w2_data[i] / 100000000;
            end
            //$display("%0.8f", w2_data[i]);
        end
        // Read layer2 bias
        $readmemh(B2_FILE, b2_memory, 0, B2_SIZE - 1);
        for(i = 0; i < B2_SIZE; i = i + 1) begin
            if(b2_memory[i] >= 134217728) begin
                b2_data[i] = b2_memory[i] - 134217728;
                b2_data[i] = b2_data[i] / 100000000 * (-1);
            end
            else begin
                b2_data[i] = b2_memory[i];
                b2_data[i] = b2_data[i] / 100000000;
            end
            //$display("%0.8f", b2_data[i]);
        end
        finish_read = 1;
        start_nn_cal = 1;
    end
    
    always begin
        #5 CLK <= ~CLK;
    end
    
    //always @(finish_read, start_nn_cal, state) begin
    always @(state, finish_read, start_nn_cal) begin
        case(state)
            0: begin
                if(finish_read == 1) begin
                    next_state = 1;
                end
                else begin
                    next_state = 0;
                end
            end
            1: begin
                if(start_nn_cal == 1) begin
                    // we use two one-dimensional array to do matrix multiplication
                    // x1 = matmul(img, layer1_weight)
                    // img.shape = (1, 784), layer1_weight.shape = (784, 128) -> x1.shape = (1, 128)
                    row1 = 1;
                    col1 = 784;
                    // row2 = 784
                    col2 = 128;
                    for (i = 0; i < row1; i = i + 1) begin
                        for (j = 0; j < col2; j = j + 1) begin
                            sum = 0.0;
                            for (k = 0; k < col1; k = k + 1)
                                sum = sum + img_data[i * col1 + k] * w1_data[k * col2 + j];
                            x1[i * col2 + j] = sum;
                        end
                    end
                    /*
                    for(i = 0; i < 128; i = i + 1) begin
                        $display("%0.8f", x1[i]);
                    end
                    */
                    next_state = 2;
                end
            end
            2: begin
                // 1. x1 = x1 + layer1_bias
                // 2. x1 = relu(x1) (relu: number becomes zero if number is negative)
                // x1.shape = (1, 128), layer1_bias.shape = (1, 128) -> x1.shape = (1, 128)
                for(i = 0; i < 128; i = i + 1) begin
                    x1[i] = x1[i] + b1_data[i];
                    if(x1[i] < 0) begin
                        x1[i] = 0;
                    end
                end
                /*
                for(i = 0; i < 128; i = i + 1) begin
                    $display("%0.8f", x1[i]);
                end
                */
                next_state = 3;
            end
            3: begin
                // x2 = matmul(x1, layer2_weight)
                // x1.shape = (1, 128), layer2_weight.shape = (128, 10) -> x2.shape = (1, 10)
                row1 = 1;
                col1 = 128;
                // row2 = 128
                col2 = 10;
                for (i = 0; i < row1; i = i + 1) begin
                    for (j = 0; j < col2; j = j + 1) begin
                        sum = 0.0;
                        for (k = 0; k < col1; k = k + 1)
                            sum = sum + x1[i * col1 + k] * w2_data[k * col2 + j];
                        x2[i * col2 + j] = sum;
                    end
                end
                /*
                for(i = 0; i < 10; i = i + 1) begin
                    $display("%0.8f", x2[i]);
                end
                */
                next_state = 4;
            end
            4: begin
                // x2 = x2 + layer2_bias
                // x2.shape = (1, 10), layer2_bias.shape = (1, 10) -> x2.shape = (1, 10)
                for(i = 0; i < 10; i = i + 1) begin
                    x2[i] = x2[i] + b2_data[i];
                end
                /*
                for(i = 0; i < 10; i = i + 1) begin
                    $display("%0.8f", x2[i]);
                end
                */
                next_state = 5;
            end
            5: begin
                max_value = -1000.0;
                predicted_digit = 0;
                for(i = 0; i < 10; i = i + 1) begin
                    if(x2[i] > max_value) begin
                        max_value = x2[i];
                        predicted_digit = i;
                    end
                end
                $display("%d", predicted_digit);
		$finish;
                next_state = 6;
            end
        endcase
    end
    
    always@(CLK) begin
        state <= next_state;
    end
    
endmodule
