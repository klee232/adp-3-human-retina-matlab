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


function [network]=rose_1_model_SVC_fine_launcher(input)
    [img_row,img_col,~]=size(input);

    %% connect input layers
    input_layer=[
      imageInputLayer([img_row img_col 3],Normalization="none",Name="input_lyr_1");
    ];
    network=dlnetwork(input_layer);


    %% connect fine stage (SRS module) (p-branch)
    % filter_size=5;
    % num_feat=filter_size*filter_size;
    fine_block_p=[
        fine_block_customConv2DLayer(3,3,256,'fine_block_p_c1','same');
        batchNormalizationLayer(Name='fine_block_p_b1');
        reluLayer(Name='fine_block_p_r1');
        fine_block_customConv2DLayer(3,256,512,'fine_block_p_c2','same');
        batchNormalizationLayer(Name='fine_block_p_b2');
        reluLayer(Name='fine_block_p_r2');
        fine_block_customConv2DLayer(3,512,9,'fine_block_p_c3','same');
        batchNormalizationLayer(Name='fine_block_p_b3');
        reluLayer(Name='fine_block_p_r3');
    ];
    network=addLayers(network,fine_block_p);
    network=connectLayers(network, 'input_lyr_1','fine_block_p_c1'); 


    %% connect fine stage (SRS module) (c-branch)
    fine_block_c=[
        fine_block_customConv2DLayer(3,512,9,'fine_block_c_c3','same');
    ];
    network=addLayers(network,fine_block_c);
    network=connectLayers(network, 'fine_block_p_r2','fine_block_c_c3'); 


    %% conduct propagation (p-branch)
    fine_block_p=[
        fine_block_input_split_layer('fine_block_p_split',2);
        fine_block_propagation_layer(2,3,'fine_block_p_p');
    ];
    network=addLayers(network,fine_block_p);
    network=connectLayers(network, 'input_lyr_1','fine_block_p_split');
    network=connectLayers(network, 'fine_block_p_a1','fine_block_p_p/in2');

    
    %% conduct propagation (c-branch)
    fine_block_c=[
        fine_block_input_split_layer('fine_block_c_split',3);
        fine_block_propagation_layer(2,3,'fine_block_c_p');
    ];
    network=addLayers(network,fine_block_c);
    network=connectLayers(network, 'input_lyr_1','fine_block_c_split');
    network=connectLayers(network, 'fine_block_c_a1','fine_block_c_p/in2');


    %% conduct fusion
    fine_block_fuse=[
        fine_block_fusion_layer(2,'fine_block_fuse');
    ];
    network=addLayers(network,fine_block_fuse);
    network=connectLayers(network, 'fine_block_p_p','fine_block_fuse/in1');
    network=connectLayers(network, 'fine_block_c_p','fine_block_fuse/in2');


end