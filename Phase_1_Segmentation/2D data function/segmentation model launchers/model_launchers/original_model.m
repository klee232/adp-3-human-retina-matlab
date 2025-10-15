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


function [train_loss_storage,valid_loss_storage,trained_net]=model(num_iteration,train_data,train_gt,valid_data,valid_gt)
    % grab out dimensional information
    [~,~,num_train_data]=size(train_data);
    [~,~,num_valid_data]=size(valid_data);

    % grab out necessary layers
    % block 1
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu 
    name="b1m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb1m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b1m1_batchnorm_layer_b";
    layerb1m1b = Batchnormalization_layer(name,1);
    name="b1m1_Relu_layer_a";
    layerb1m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b1m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb1m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b1m2_batchnorm_layer_b";
    layerb1m2b = Batchnormalization_layer(name,1);
    name="b1m2_Relu_layer_a";
    layerb1m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b1m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb1m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b1m3_batchnorm_layer_b";
    layerb1m3b = Batchnormalization_layer(name,1);
    name="b1m3_Relu_layer_a";
    layerb1m3a = ReLu_layer(name);
    % Residual module layer 1: Residual layer
    name="rml1_Residual_layer";
    layerrm1=Residual_layer(name);
    name="rml1_batchnorm_layer_b";
    layerrm1b = Batchnormalization_layer(name,1);
    name="rml1_Relu_layer_a";
    layerrm1a = ReLu_layer(name);

    % block 2
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu
    name="b2m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb2m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b2m1_batchnorm_layer_b";
    layerb2m1b = Batchnormalization_layer(name,1);
    name="b2m1_Relu_layer_a";
    layerb2m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b2m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb2m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b2m2_batchnorm_layer_b";
    layerb2m2b = Batchnormalization_layer(name,1);
    name="b2m2_Relu_layer_a";
    layerb2m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b2m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb2m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b2m3_batchnorm_layer_b";
    layerb2m3b = Batchnormalization_layer(name,1);
    name="b2m3_Relu_layer_a";
    layerb2m3a = ReLu_layer(name);
    % Residual module layer 2: Residual layer
    name="rml2_Residual_layer";
    layerrm2=Residual_layer(name);
    name="rml2_batchnorm_layer_b";
    layerrm2b = Batchnormalization_layer(name,1);
    name="rml2_Relu_layer_a";
    layerrm2a = ReLu_layer(name);

    % block 3
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu
    name="b3m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb3m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b3m1_batchnorm_layer_b";
    layerb3m1b = Batchnormalization_layer(name,1);
    name="b3m1_Relu_layer_a";
    layerb3m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b3m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb3m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b3m2_batchnorm_layer_b";
    layerb3m2b = Batchnormalization_layer(name,1);
    name="b3m2_Relu_layer_a";
    layerb3m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b3m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb3m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b3m3_batchnorm_layer_b";
    layerb3m3b = Batchnormalization_layer(name,1);
    name="b3m3_Relu_layer_a";
    layerb3m3a = ReLu_layer(name);
    % Residual module layer 3: Residual layer
    name="rml3_Residual_layer";
    layerrm3=Residual_layer(name);
    name="rml3_batchnorm_layer_b";
    layerrm3b = Batchnormalization_layer(name,1);
    name="rml3_Relu_layer_a";
    layerrm3a = ReLu_layer(name);

    % block 4 Decoder 
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu
    name="b4m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb4m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b4m1_batchnorm_layer_b";
    layerb4m1b = Batchnormalization_layer(name,1);
    name="b4m1_Relu_layer_a";
    layerb4m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b4m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb4m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b4m2_batchnorm_layer_b";
    layerb4m2b = Batchnormalization_layer(name,1);
    name="b4m2_Relu_layer_a";
    layerb4m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b4m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb4m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b4m3_batchnorm_layer_b";
    layerb4m3b = Batchnormalization_layer(name,1);
    name="b4m3_Relu_layer_a";
    layerb4m3a = ReLu_layer(name);
    % Residual module layer 4: Residual layer
    name="rml4_Residual_layer";
    layerrm4=Residual_layer(name);
    name="rml4_batchnorm_layer_b";
    layerrm4b = Batchnormalization_layer(name,1);
    name="rml4_Relu_layer_a";
    layerrm4a = ReLu_layer(name);

    % Concatenate layer
    name="concat_l_1";
    concate_dim=3;
    num_data=num_train_data;
    layer_c1 = Concatenate_layer(name,concate_dim,num_data);
    num_data=num_valid_data;
    layer_c1_v = Concatenate_layer(name,concate_dim,num_data);

    % block 5 Decoder 
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu
    name="b5m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=2;
    layerb5m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b5m1_batchnorm_layer_b";
    layerb5m1b = Batchnormalization_layer(name,1);
    name="b5m1_Relu_layer_a";
    layerb5m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b5m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb5m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b5m2_batchnorm_layer_b";
    layerb5m2b = Batchnormalization_layer(name,1);
    name="b5m2_Relu_layer_a";
    layerb5m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b5m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb5m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b5m3_batchnorm_layer_b";
    layerb5m3b = Batchnormalization_layer(name,1);
    name="b5m3_Relu_layer_a";
    layerb5m3a = ReLu_layer(name);
    % Residual module layer 5: Residual layer
    name="rml5_Residual_layer";
    layerrm5=Residual_layer(name);
    name="rml5_batchnorm_layer_b";
    layerrm5b = Batchnormalization_layer(name,1);
    name="rml5_Relu_layer_a";
    layerrm5a = ReLu_layer(name);

    % Concatenate layer
    name="concat_l_2";
    concate_dim=3;
    num_data=num_train_data;
    layer_c2 = Concatenate_layer(name,concate_dim,num_data);
    num_data=num_valid_data;
    layer_c2_v = Concatenate_layer(name,concate_dim,num_data);

    % block 6 Decoder 
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu
    name="b6m1_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=2;
    layerb6m1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b6m1_batchnorm_layer_b";
    layerb6m1b = Batchnormalization_layer(name,1);
    name="b6m1_Relu_layer_a";
    layerb6m1a = ReLu_layer(name);
    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    name="b6m2_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb6m2 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b6m2_batchnorm_layer_b";
    layerb6m2b = Batchnormalization_layer(name,1);
    name="b6m2_Relu_layer_a";
    layerb6m2a = ReLu_layer(name);
    % module layer 3: 3*3 convolutional layer + batchnromalization + relu
    name="b6m3_conv_layer";
    num_filter=1;
    kernel_size=3;
    input_chn=1;
    layerb6m3 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="b6m3_batchnorm_layer_b";
    layerb6m3b = Batchnormalization_layer(name,1);
    name="b6m3_Relu_layer_a";
    layerb6m3a = ReLu_layer(name);
    % Residual module layer 6: Residual layer
    name="rml6_Residual_layer";
    layerrm6=Residual_layer(name);
    name="rml6_batchnorm_layer_b";
    layerrm6b = Batchnormalization_layer(name,1);
    name="rml6_Relu_layer_a";
    layerrm6a = ReLu_layer(name);

    % block 7 flatten layer
    name="f1_conv_layer";
    num_filter=1;
    kernel_size=1;
    input_chn=1;
    layerf1 = Convolutional_2D_layer(name,num_filter,kernel_size,input_chn,"same");
    name="f1_batchnorm_layer_b";
    layerf1b = Batchnormalization_layer(name,1);
    name="f1_Relu_layer_a";
    layerf1a = Sigmoid_layer(name);

    % block 8 loss function
    name="SoftDice";
    layer_l=SoftDice_ClassificationLayer(name);

    % save trained network
    trained_net=[layerb1m1 layerb1m1b layerb1m1a...
                 layerb1m2 layerb1m2b layerb1m2a ...
                 layerb1m3 layerb1m3b layerb1m3a ...
                 layerrm1 layerrm1b layerrm1a ...
                 layerb2m1 layerb2m1b layerb2m1a...
                 layerb2m2 layerb2m2b layerb2m2a ...
                 layerb2m3 layerb2m3b layerb2m3a ...
                 layerrm2 layerrm2b layerrm2a ...
                 layerb3m1 layerb3m1b layerb3m1a...
                 layerb3m2 layerb3m2b layerb3m2a ...
                 layerb3m3 layerb3m3b layerb3m3a ...
                 layerrm3 layerrm3b layerrm3a ...
                 layerb4m1 layerb4m1b layerb4m1a...
                 layerb4m2 layerb4m2b layerb4m2a ...
                 layerb4m3 layerb4m3b layerb4m3a ...
                 layerrm4 layerrm4b layerrm4a ...
                 layer_c1 ...
                 layerb5m1 layerb5m1b layerb5m1a...
                 layerb5m2 layerb5m2b layerb5m2a ...
                 layerb5m3 layerb5m3b layerb5m3a ...
                 layerrm5 layerrm5b layerrm5a ...          
                 layer_c2 ...
                 layerb6m1 layerb6m1b layerb6m1a...
                 layerb6m2 layerb6m2b layerb6m2a ...
                 layerb6m3 layerb6m3b layerb6m3a ...
                 layerrm6 layerrm6b layerrm6a ... 
                 layerf1 layerf1b layerf1a ...
                 layer_l];

end