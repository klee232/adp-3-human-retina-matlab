% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter
% kernel_size: size of the convolutional kernel (integer)
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)


function [train_loss_storage,valid_loss_storage,...
          W1_Progress,B1_Progress,...
          W3_Progress,B3_Progress,...
          W4_Progress,B4_Progress,...
          W6_Progress,B6_Progress]=SegVess_model(num_iteration,train_data,train_gt,valid_data,valid_gt)

    % grab out necessary layers
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu 
    name="m1_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm1 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m1_batchnorm_layer_b";
    layerm1b = Batchnormalization_layer(name,1);
    name="m1_Relu_layer_a";
    layerm1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="m2_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm2 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m2_batchnorm_layer_b";
    layerm2b = Batchnormalization_layer(name,1);
    name="m2_Relu_layer_a";
    layerm2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="m3_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm3 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m3_batchnorm_layer_b";
    layerm3b = Batchnormalization_layer(name,1);
    name="m3_Relu_layer_a";
    layerm3a = ReLu_layer(name);
    % layer 4: Residual layer
    name="l4_Residual_layer";
    layer4=Residual_layer(name);
    % module layer 5: 3*3 convolutional layer + batchnromalization + relu 
    name="m5_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm5 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m5_batchnorm_layer_b";
    layerm5b = Batchnormalization_layer(name,1);
    name="m5_Relu_layer_a";
    layerm5a = ReLu_layer(name);
    % module layer 6: 3*3 convolutional layer + batchnromalization + relu 
    name="m6_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm6 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m6_batchnorm_layer_b";
    layerm6b = Batchnormalization_layer(name,1);
    name="m6_Relu_layer_a";
    layerm6a = ReLu_layer(name);
    % module layer 7: 3*3 convolutional layer + batchnromalization + relu 
    name="m7_conv_layer";
    kernel_size=3;
    kernel_feat=1;
    layerm7 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="m7_batchnorm_layer_b";
    layerm7b = Batchnormalization_layer(name,1);
    name="m7_Relu_layer_a";
    layerm7a = ReLu_layer(name);




   
    % layer 3: 3*3 convolutional layer
    name="conv_layer3";
    kernel_size=3;
    kernel_feat=1;
    layer3 = Convolutional_2D_layer(name,kernel_size,kernel_feat,1,"same");
    name="batchnorm_layer_b3";
    layer3b = Batchnormalization_layer(name,1);
    name="Relu_layer_a3";
    layer3a = ReLu_layer(name);
    % layer 4: transpose convolution
    name="transconv_layer4";
    kernel_size=2;
    layer4 = Transposeconvolutional_2D_layer(name,kernel_size,1);
    name="batchnorm_layer_b4";
    layer4b = Batchnormalization_layer(name,1);
    name="Relu_layer_a4";
    layer4a = ReLu_layer(name);
    % layer 6: 3*3 convolutional layer
    name="conv_layer6";
    kernel_size=3;
    kernel_size_feat=1;
    layer6 = Convolutional_2D_layer(name,kernel_size,kernel_size_feat,1,"same");
    name="batchnorm_layer_b6";
    layer6b = Batchnormalization_layer(name,1);
    name="Relu_layer_a6";
    layer6a = Sigmoid_layer(name);
    % layer 8: loss function 
    name="softdice_layer8";
    layer8=SoftDice_ClassificationLayer(name);

    % create storage variable for storing training and validation loss
    train_loss_storage=zeros(num_iteration,1);
    valid_loss_storage=zeros(num_iteration,1);

    % create storage variable for storing samples from weights and biases
    W6_Progress=zeros(num_iteration,1);
    B6_Progress=zeros(num_iteration,1);
    W4_Progress=zeros(num_iteration,1);
    B4_Progress=zeros(num_iteration,1);
    W3_Progress=zeros(num_iteration,1);
    B3_Progress=zeros(num_iteration,1);
    W1_Progress=zeros(num_iteration,1);
    B1_Progress=zeros(num_iteration,1);

    tic
    % conduct training for the model
    % update current parameters
    % layer 6
    momentum6W_1=0;
    momentum6W_2=0;
    momentum6B_1=0;
    momentum6B_2=0;
    momentum6b_1=0;
    momentum6b_2=0;
    momentum6r_1=0;
    momentum6r_2=0;
    % layer 4
    momentum4W_1=0;
    momentum4W_2=0;
    momentum4B_1=0;
    momentum4B_2=0;
    momentum4b_1=0;
    momentum4b_2=0;
    momentum4r_1=0;
    momentum4r_2=0;
    % layer 3
    momentum3W_1=0;
    momentum3W_2=0;
    momentum3B_1=0;
    momentum3B_2=0;
    momentum3b_1=0;
    momentum3b_2=0;
    momentum3r_1=0;
    momentum3r_2=0;
    % layer 1
    momentum1W_1=0;
    momentum1W_2=0;
    momentum1B_1=0;
    momentum1B_2=0;
    momentum1b_1=0;
    momentum1b_2=0;
    momentum1r_1=0;
    momentum1r_2=0;
    for i_iter=1:num_iteration
        % forward part
        % training data part
        forward1=predict(layer1,train_data); % 3*3 convolutional layer
        forward1b=predict(layer1b,forward1); % batchnormalization
        forward1a=predict(layer1a,forward1b); % ReLu layer

        forward1t=predict(layer1,forward1a); % 3*3 convolutional layer
        forward1bt=predict(layer1b,forward1t); % batchnormalization
        forward1at=predict(layer1a,forward1bt); % ReLu layer

        forward1tt=predict(layer1,forward1at); % 3*3 convolutional layer
        forward1btt=predict(layer1b,forward1tt); % batchnormalization
        forward1att=predict(layer1a,forward1btt); % ReLu layer

        [out1, out2]=predict(layer4,forward1att);


        [forward2,max_mask2]=predict(layer2,forward1a); % 2*2 maxpooling layer
        forward3=predict(layer3,forward2); % 3*3 convolutional layer
        forward3b=predict(layer3b,forward3); % batchnormalization
        forward3a=predict(layer3a,forward3b); % ReLu layer
        forward4=predict(layer4,forward3a); % 3*3 transposed convolutional layer
        forward4b=predict(layer4b,forward4); % batchnormalization
        forward4a=predict(layer4a,forward4b); % ReLu layer
        forward6=predict(layer6,forward4a); % 3*3*2 convolutional layer
        forward6b=predict(layer6b,forward6); % batchnormalization
        forward6a=predict(layer6a,forward6b); % Sigmoid layer (Change it to Sigmoid)
        forward8=forwardLoss(layer8,forward6a,train_gt); % soft dice loss function
        % display the current training loss
        message=strcat("Current iteration ", string(i_iter), " out of ", string(num_iteration)," : ", string(forward8));
        disp(message);
        % store current loss value for plotting the loss plot
        train_loss_storage(i_iter,1)=forward8;
        % validation data part
        forward1_v=predict(layer1,valid_data); % 3*3 convolutional layer
        forward1b_v=predict(layer1b,forward1_v); % batchnormalization
        forward1a_v=predict(layer1a,forward1b_v); % ReLu layer
        forward2_v=predict(layer2,forward1a_v); % 2*2 maxpooling layer
        forward3_v=predict(layer3,forward2_v); % 3*3 convolutional layer
        forward3b_v=predict(layer3b,forward3_v); % batchnormalization
        forward3a_v=predict(layer3a,forward3b_v); % ReLu layer
        forward4_v=predict(layer4,forward3a_v); % 3*3 transposed convolutional layer
        forward4b_v=predict(layer4b,forward4_v); % batchnormalization
        forward4a_v=predict(layer4a,forward4b_v); % ReLu layer
        forward6_v=predict(layer6,forward4a_v); % 3*3 convolutional layer
        forward6b_v=predict(layer6b,forward6_v); % batchnormalization
        [forward6a_v]=predict(layer6a,forward6b_v); % ReLu layer
        forward8_v=forwardLoss(layer8,forward6a_v,valid_gt); % soft dice loss function (need modification)
        % display the current training loss
        message=strcat("Current iteration (Validation) ", string(i_iter), " out of ", string(num_iteration)," : ", string(forward8_v));
        disp(message);
        % store current loss value for plotting the loss plot
        valid_loss_storage(i_iter,1)=forward8_v;

        % backward part
        dLdY_8=backwardLoss(layer8,forward6a,train_gt); % soft dice loss function
        dLdY_6a=backward(layer6a,forward6b,dLdY_8); % ReLu layer
        [dLdY_6b,dLdr_6,dLdb_6] = backward(layer6b,forward6,dLdY_6a); % batchnormalization
        [dLdY_6,dLdW_6,dLdB_6] = backward(layer6,forward4a,dLdY_6b); % 3*3 convolutional layer
        dLdY_4a=backward(layer4a,forward4b,dLdY_6); % ReLu layer
        [dLdY_4b,dLdr_4,dLdb_4] = backward(layer4b,forward4,dLdY_4a); % batchnormalization
        [dLdY_4,dLdW_4,dLdB_4] = backward(layer4,forward3a,dLdY_4b); % 3*3 transposed convolutional layer
        dLdY_3a=backward(layer3a,forward3b,dLdY_4); % ReLu layer
        [dLdY_3b,dLdr_3,dLdb_3] = backward(layer3b,forward3,dLdY_3a); % batchnormalization
        [dLdY_3,dLdW_3,dLdB_3] = backward(layer3,forward2,dLdY_3b); % 3*3 convolutional layer
        [dLdY_2] = backward(layer2,max_mask2,dLdY_3); % 2*2 maxpooling layer
        dLdY_1a=backward(layer1a,forward1b,dLdY_2);% ReLu
        [dLdY_1b,dLdr_1,dLdb_1] = backward(layer1b,forward1,dLdY_1a); % batchnormalization
        [dLdY_1,dLdW_1,dLdB_1] = backward(layer1,train_data,dLdY_1b);% 3*3 cconvolutional layer
        % optimizing the parameters 
        % layer 6
        [updated_W_6,out_momentum1_W_6,out_momentum2_W_6]=Adam_Optimizer(momentum6W_1,momentum6W_2,0.9,0.999,0.01,i_iter,layer6.Weights,dLdW_6);
        [updated_B_6,out_momentum1_B_6,out_momentum2_B_6]=Adam_Optimizer(momentum6B_1,momentum6B_2,0.9,0.999,0.01,i_iter,layer6.Bias,dLdB_6);
        [updated_b_6,out_momentum1_b_6,out_momentum2_b_6]=Adam_Optimizer(momentum6b_1,momentum6b_2,0.9,0.999,0.01,i_iter,layer6b.beta,dLdb_6);
        [updated_r_6,out_momentum1_r_6,out_momentum2_r_6]=Adam_Optimizer(momentum6r_1,momentum6r_2,0.9,0.999,0.01,i_iter,layer6b.gamma,dLdr_6);
        % layer 4
        [updated_W_4,out_momentum1_W_4,out_momentum2_W_4]=Adam_Optimizer(momentum4W_1,momentum4W_2,0.9,0.999,0.01,i_iter,layer4.Weights,dLdW_4);
        [updated_B_4,out_momentum1_B_4,out_momentum2_B_4]=Adam_Optimizer(momentum4B_1,momentum4B_2,0.9,0.999,0.01,i_iter,layer4.Bias,dLdB_4);
        [updated_b_4,out_momentum1_b_4,out_momentum2_b_4]=Adam_Optimizer(momentum4b_1,momentum4b_2,0.9,0.999,0.01,i_iter,layer4b.beta,dLdb_4);
        [updated_r_4,out_momentum1_r_4,out_momentum2_r_4]=Adam_Optimizer(momentum4r_1,momentum4r_2,0.9,0.999,0.01,i_iter,layer4b.gamma,dLdr_4);
        % layer 3
        [updated_W_3,out_momentum1_W_3,out_momentum2_W_3]=Adam_Optimizer(momentum3W_1,momentum3W_2,0.9,0.999,0.01,i_iter,layer3.Weights,dLdW_3);
        [updated_B_3,out_momentum1_B_3,out_momentum2_B_3]=Adam_Optimizer(momentum3B_1,momentum3B_2,0.9,0.999,0.01,i_iter,layer3.Bias,dLdB_3);
        [updated_b_3,out_momentum1_b_3,out_momentum2_b_3]=Adam_Optimizer(momentum3b_1,momentum3b_2,0.9,0.999,0.01,i_iter,layer3b.beta,dLdb_3);
        [updated_r_3,out_momentum1_r_3,out_momentum2_r_3]=Adam_Optimizer(momentum3r_1,momentum3r_2,0.9,0.999,0.01,i_iter,layer3b.gamma,dLdr_3);
         % layer 1
        [updated_W_1,out_momentum1_W_1,out_momentum2_W_1]=Adam_Optimizer(momentum1W_1,momentum1W_2,0.9,0.999,0.01,i_iter,layer1.Weights,dLdW_1);
        [updated_B_1,out_momentum1_B_1,out_momentum2_B_1]=Adam_Optimizer(momentum1B_1,momentum1B_2,0.9,0.999,0.01,i_iter,layer1.Bias,dLdB_1);
        [updated_b_1,out_momentum1_b_1,out_momentum2_b_1]=Adam_Optimizer(momentum1b_1,momentum1b_2,0.9,0.999,0.01,i_iter,layer1b.beta,dLdb_1);
        [updated_r_1,out_momentum1_r_1,out_momentum2_r_1]=Adam_Optimizer(momentum1r_1,momentum1r_2,0.9,0.999,0.01,i_iter,layer1b.gamma,dLdr_1);
        prev=layer1.Weights;
        % store a sample point for weight and bias parameters
        % layer 6
        W6=layer6.Weights;
        B6=layer6.Bias;
        [row_W6,col_W6]=size(W6);
        W6_Progress(i_iter,1)=W6(ceil(row_W6/2),ceil(col_W6/2));
        B6_Progress(i_iter,1)=B6;
        % layer 4
        W4=layer4.Weights;
        B4=layer4.Bias;
        [row_W4,col_W4]=size(W4);
        W4_Progress(i_iter,1)=W4(ceil(row_W4/2),ceil(col_W4/2));
        B4_Progress(i_iter,1)=B4;
        % layer 3
        W3=layer3.Weights;
        B3=layer3.Bias;
        [row_W3,col_W3]=size(W3);
        W3_Progress(i_iter,1)=W3(ceil(row_W3/2),ceil(col_W3/2));
        B3_Progress(i_iter,1)=B3;
        % layer 1
        W1=layer1.Weights;
        B1=layer1.Bias;
        [row_W1,col_W1]=size(W1);
        W1_Progress(i_iter,1)=W1(ceil(row_W1/2),ceil(col_W1/2));
        B1_Progress(i_iter,1)=B1;


        % update current parameters
        % layer 6
        layer6.Weights=updated_W_6; 
        layer6.Bias=updated_B_6;
        layer6b.beta=updated_b_6;
        layer6b.gamma=updated_r_6;
        momentum6W_1=out_momentum1_W_6;
        momentum6W_2=out_momentum2_W_6;
        momentum6B_1=out_momentum1_B_6;
        momentum6B_2=out_momentum2_B_6;
        momentum6b_1=out_momentum1_b_6;
        momentum6b_2=out_momentum2_b_6;
        momentum6r_1=out_momentum1_r_6;
        momentum6r_2=out_momentum2_r_6;
        % layer 4
        layer4.Weights=updated_W_4; 
        layer4.Bias=updated_B_4;
        layer4b.beta=updated_b_4;
        layer4b.gamma=updated_r_4;
        momentum4W_1=out_momentum1_W_4;
        momentum4W_2=out_momentum2_W_4;
        momentum4B_1=out_momentum1_B_4;
        momentum4B_2=out_momentum2_B_4;
        momentum4b_1=out_momentum1_b_4;
        momentum4b_2=out_momentum2_b_4;
        momentum4r_1=out_momentum1_r_4;
        momentum4r_2=out_momentum2_r_4;
        % layer 3
        layer3.Weights=updated_W_3; 
        layer3.Bias=updated_B_3;
        layer3b.beta=updated_b_3;
        layer3b.gamma=updated_r_3;
        momentum3W_1=out_momentum1_W_3;
        momentum3W_2=out_momentum2_W_3;
        momentum3B_1=out_momentum1_B_3;
        momentum3B_2=out_momentum2_B_3;
        momentum3b_1=out_momentum1_b_3;
        momentum3b_2=out_momentum2_b_3;
        momentum3r_1=out_momentum1_r_3;
        momentum3r_2=out_momentum2_r_3;
        % layer 1
        layer1.Weights=updated_W_1; 
        layer1.Bias=updated_B_1;
        layer1b.beta=updated_b_1;
        layer1b.gamma=updated_r_1;
        momentum1W_1=out_momentum1_W_1;
        momentum1W_2=out_momentum2_W_1;
        momentum1B_1=out_momentum1_B_1;
        momentum1B_2=out_momentum2_B_1;
        momentum1b_1=out_momentum1_b_1;
        momentum1b_2=out_momentum2_b_1;
        momentum1r_1=out_momentum1_r_1;
        momentum1r_2=out_momentum2_r_1;
    end
    toc

    % save trained network
    trained_net=[layer1...
                 layer1b...
                 layer1a...
                 layer2...
                 layer3...
                 layer3b...
                 layer3a...
                 layer4...
                 layer4b...
                 layer4a...
                 layer6...
                 layer6b...
                 layer6a...
                 layer8];
    save trained_net


end