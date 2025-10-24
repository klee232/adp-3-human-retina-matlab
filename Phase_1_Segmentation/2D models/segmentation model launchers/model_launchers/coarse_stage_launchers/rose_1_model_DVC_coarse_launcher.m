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


function [network]=rose_1_model_DVC_coarse_launcher(input)
    [img_row,img_col,~]=size(input);
    

    %% setup input layer 
    num_feat=64;
    layerinput_b=[
        imageInputLayer([img_row img_col 1],Normalization="none",Name="input_lyr");
        convolution2dLayer([3 3],num_feat,Stride=2,Padding=[1 1],Name='input_lyr_c1');
        batchNormalizationLayer(Name='input_lyr_b1');
        reluLayer(Name='input_lyr_r1');
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c2');
        batchNormalizationLayer(Name='input_lyr_b2');
        reluLayer(Name='input_lyr_r2');
        convolution2dLayer([3 3],num_feat*2,Padding="same",Name='input_lyr_c3');
        batchNormalizationLayer(Name='input_lyr_b3');
        reluLayer(Name='input_lyr_r3');
        maxPooling2dLayer([3 3],Stride=[2 2], Padding=[1 1],Name='input_lyr_max');
    ];
    network=dlnetwork(layerinput_b);   


    %% connect coarse stage (p branch)
    % encoder part
    num_cardinal=2;
    radix=2;
    num_layer=1;
    num_feat=4*64;
    num_block=3;
    % layer 1: (76*76), layer 2: (38*38), layer 3: (19*19), layer 4:
    % (19*19)
    for i_layer=1:num_layer
        % grab out current block number for current layer
        current_num_block=num_block(i_layer);
        current_num_feat=num_feat(i_layer);
        % if this is the first layer
        for i_block=1:current_num_block
            % initiate bottleneck block
            [network]=res_block_bottleneck(network, i_layer, i_block, current_num_feat, radix, num_cardinal);
            % connect the input
            if i_block==1
                if i_layer==1
                    network=connectLayers(network, 'input_lyr_max', strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                elseif i_layer==2
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer-1),"_",string(num_block(i_layer-1))), strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                else
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))), strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                end
            else
                network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block-1)), strcat('main_layer_c1_',string(i_layer),"_",string(i_block)));
            end
        end

    end


    %% connect coarse stage (c branch)
    % decoder part
    for i_layer=(num_layer):(-1):1
         current_num_feat=num_feat(i_layer);
         % if the current layer is 4, connect the bottleneck and encoder layer 4 to the concatenation layer
         res_dblock_transcat=[
             transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_d_transcat_t_',string(i_layer)));
             res_block_pad_cat_layer(strcat('res_dblock_transcat_cat_',string(i_layer)),2);
         ];
         [network]=addLayers(network,res_dblock_transcat);
         network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block)),strcat('res_dblock_d_transcat_t_',string(i_layer)));
         network=connectLayers(network, 'input_lyr_r1',strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));

         [network]=res_dblock_bottleneck_d(network, i_layer, 1, current_num_feat, radix, num_cardinal);
         network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(1)));
         network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(1)));
    end

    layer_flatten_coarse=[
          transposedConv2dLayer([2 2],1,"Stride",[2 2],Name=strcat('layer_flatten_coarse_c_t1'));
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_c_c1');
          sigmoidLayer(Name='layer_flatten_coarse_c_a1');
    ];
    
    network=addLayers(network,layer_flatten_coarse);
    network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(1)),'layer_flatten_coarse_c_t1');

end