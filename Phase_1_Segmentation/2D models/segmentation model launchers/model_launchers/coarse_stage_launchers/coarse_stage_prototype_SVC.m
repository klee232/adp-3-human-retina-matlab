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


function [network]=coarse_stage_prototype_SVC(input)
    [img_row,img_col,~]=size(input);
   
    
    %% setup input layer (conv1 in the original model) (checked)
    num_feat=64;
    layerinput_b=[
        imageInputLayer([img_row img_col 1],Normalization="none",Name="input_lyr");
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c1');
        batchNormalizationLayer(Name='input_lyr_b1');
        reluLayer(Name='input_lyr_r1'); % feature map size 304*304*64
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c2');
        batchNormalizationLayer(Name='input_lyr_b2');
        reluLayer(Name='input_lyr_r2'); % feature map size 304*304*64
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c3');
        batchNormalizationLayer(Name='input_lyr_b3');
        reluLayer(Name='input_lyr_r3'); % feature map size 304*304*64
        % maxPooling2dLayer([2 2],Stride=[2 2], Name='input_lyr_max'); % feature map size 152*152*128
        ];
    network=dlnetwork(layerinput_b); 


    %% connect coarse stage (p branch)
    % encoder part
    num_cardinal=2;
    radix=2;
    num_layer=5;
    num_feat=[128 256 512 1024 2048];
    num_block=[3 4 4 6 6];
    % layer 1: (304*304), layer 2: (152*152), layer 3: (76*76), layer 4:
    % (38*38), layer 5 (bottle neck):(19*19)
    for i_layer=1:num_layer
        % grab out current block number for current layer
        current_num_block=num_block(i_layer);
        current_num_feat=num_feat(i_layer);
        % if this is the first layer
        for i_block=1:current_num_block
            % initiate bottleneck block
            [network]=res_block_bottleneck(network, i_layer, i_block, current_num_feat, radix, num_cardinal);
            % connect the input
            if i_layer==1
                if i_block==1
                    network=connectLayers(network, 'input_lyr_r3', strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));                    
                else
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block-1)), strcat('main_layer_c1_',string(i_layer),"_",string(i_block)));
                end
            else
                if i_block==1
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))), strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                else
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block-1)), strcat('main_layer_c1_',string(i_layer),"_",string(i_block)));
                end
            end
            if i_block==current_num_block && i_layer~=5 % insert max pooling to all layers (except for layer 5 (bottleneck))
                current_lyr_maxPool=[
                    maxPooling2dLayer([2 2],Stride=[2 2],Name=strcat('current_lyr_max_',string(i_layer),'_',string(i_block)));
                    ];
                [network]=addLayers(network,current_lyr_maxPool);
                network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block)),strcat('current_lyr_max_',string(i_layer),'_',string(i_block)));
            end
        end

    end


    % decoder part
    % layer 4: (38,38) (4 and 3), layer 3: (76,76) (3 and 2) , layer 2 (152,152) (2 and 1) 
    % grab out current number of features and blocks
    num_layer=4;
    for i_layer=num_layer:(-1):1
        current_num_feat=num_feat(i_layer);
        current_num_block=num_block(i_layer);

        % loop throught layer
        for i_block=1:current_num_block
            % create transposed and concatenation block if this is the
            % first block
            if i_block==1
                res_dblock_transcat=[
                    transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_transcat_t_',string(i_layer)));
                    res_block_pad_cat_layer(strcat('res_dblock_transcat_cat_',string(i_layer)),2);
                    ];
                [network]=addLayers(network,res_dblock_transcat);
                if i_layer==4
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer+1),"_",string(num_block(i_layer+1))),strcat('res_dblock_transcat_t_',string(i_layer)));
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(num_block(i_layer))),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
                else
                    network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer+1), "_", string(num_block(i_layer+1))),strcat('res_dblock_transcat_t_',string(i_layer)));
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(num_block(i_layer))),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
                end
            end
            if i_layer==4 && i_block==1 % checked
                res_dblock_drop=[
                    res_dblock_spatial_dropout_layer('res_dblock_drop',0.2);
                    ];
                [network]=addLayers(network,res_dblock_drop);
                network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)), 'res_dblock_drop');
            end
            [network]=res_dblock_bottleneck(network, i_layer, i_block, current_num_feat, radix, num_cardinal);

            if i_layer==4 && i_block==1
                network=connectLayers(network, 'res_dblock_drop',strcat('d_main_layer_c1_',string(i_layer),'_',string(1)));
                network=connectLayers(network, 'res_dblock_drop', strcat('d_residual_layer_c1_',string(i_layer), "_", string(1)));
            else
                if i_block==1
                    network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)),strcat('d_main_layer_c1_',string(i_layer),'_',string(i_block)));
                    network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)), strcat('d_residual_layer_c1_',string(i_layer), "_", string(i_block)));
                else
                    network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer), "_", string(i_block-1)),strcat('d_main_layer_c1_',string(i_layer),'_',string(i_block)));
                    network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer), "_", string(i_block-1)), strcat('d_residual_layer_c1_',string(i_layer), "_", string(i_block)));
                end
            end
        end

    end

    layer_flatten_coarse_p=[
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_p_c1');
          sigmoidLayer(Name='layer_flatten_coarse_p_a1');
    ];
    network=addLayers(network,layer_flatten_coarse_p);
    network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer), "_", string(num_block(i_layer))),'layer_flatten_coarse_p_c1');


    %% connect coarse stage (c branch)
    % decoder part
    for i_layer=1:1     
         current_num_feat=num_feat(i_layer);
         current_num_block=num_block(i_layer);

         % create transpose and concatenation layer

         res_dblock_d_transcat=[
             transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_d_transcat_t_',string(i_layer)));
             res_block_pad_cat_layer(strcat('res_dblock_d_transcat_cat_',string(i_layer)),2);
             res_dblock_spatial_dropout_layer(strcat('res_dblock_d_drop_',string(i_layer)),0.2);
             ];
         [network]=addLayers(network,res_dblock_d_transcat);
         network=connectLayers(network, strcat('res_block_final_add_',string(i_layer+1),"_",string(num_block(i_layer+1))),strcat('res_dblock_d_transcat_t_',string(i_layer)));
         network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(num_block(i_layer))),strcat('res_dblock_d_transcat_cat_',string(i_layer),"/in2"));
         
         for i_block=1:current_num_block
             [network]=res_dblock_bottleneck_d(network, i_layer, i_block, current_num_feat, radix, num_cardinal);
             if i_block==1
                network=connectLayers(network, strcat('res_dblock_d_drop_',string(i_layer)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(i_block)));
                network=connectLayers(network, strcat('res_dblock_d_drop_',string(i_layer)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(i_block)));
             else
                network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(i_block-1)),strcat('d_main_layer_d_c1_',string(i_layer),'_',string(i_block)));
                network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(i_block-1)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(i_block)));
             end
         end

    end

    layer_flatten_coarse_c=[
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_c_c1');
          sigmoidLayer(Name='layer_flatten_coarse_c_a1');
    ];
    network=addLayers(network,layer_flatten_coarse_c);
    network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(num_block(i_layer))),'layer_flatten_coarse_c_c1');


    %% output layer
    layer_output_coarse=[
        coarse_block_output_conatenate_layer('layer_output_coarse_cat',2);
    ];
    network=addLayers(network,layer_output_coarse);
    network=connectLayers(network, 'layer_flatten_coarse_p_a1',"layer_output_coarse_cat/in1");
    network=connectLayers(network, 'layer_flatten_coarse_c_a1',"layer_output_coarse_cat/in2");



end