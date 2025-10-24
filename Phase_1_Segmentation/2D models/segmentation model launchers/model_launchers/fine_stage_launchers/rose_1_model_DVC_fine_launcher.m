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


function [network]=rose_1_model_DVC_fine_launcher(input)
    [img_row,img_col,~]=size(input);

    %% connect input layers
    input_layer_1=[
      imageInputLayer([img_row img_col 2],Normalization="none",Name="input_lyr_1");
    ];
    network=dlnetwork(input_layer_1);


    %% connect fine stage (SRS module) (p-branch)
    % filter_size=5;
    % num_feat=filter_size*filter_size;
    fine_block=[
        fine_block_customConv2DLayer(3,3,256,'fine_block_c1','same');
        batchNormalizationLayer(Name='fine_block_b1');
        reluLayer(Name='fine_block_r1');
        fine_block_customConv2DLayer(3,256,512,'fine_block_c2','same');
        batchNormalizationLayer(Name='fine_block_b2');
        reluLayer(Name='fine_block_r2');
        fine_block_customConv2DLayer(5,512,25,'fine_block_c3','same');
        batchNormalizationLayer(Name='fine_block_b3');
        reluLayer(Name='fine_block_r3');
        fine_block_softmax_layer('fine_block_a1')
    ];
    network=addLayers(network,fine_block);
    network=connectLayers(network, 'input_lyr_1','fine_block_c1');


    %% connect fine stage (SRS module) (c-branch) 
    fine_block_input_split=[
        fine_block_input_split_layer('fine_block_input_split',2);
    ];
    network=addLayers(network,fine_block_input_split);
    network=connectLayers(network, 'input_lyr_1','fine_block_input_split');

    fine_block=[
        fine_block_propagation_layer(2,5,'fine_block');
    ];
    network=addLayers(network,fine_block);
    network=connectLayers(network, 'fine_block_input_split','fine_block/in1');
    network=connectLayers(network, 'fine_block_a1','fine_block/in2');


end