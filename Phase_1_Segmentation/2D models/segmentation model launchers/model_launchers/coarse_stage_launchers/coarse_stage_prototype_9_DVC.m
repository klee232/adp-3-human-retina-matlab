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


function [network]=coarse_stage_prototype_9_DVC(input)
    [img_row,img_col,~]=size(input);
    

    %% setup input layer 
    num_feat=64;
    layerinput_b=[
        imageInputLayer([img_row img_col 1],Normalization="none",Name="input_lyr");
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c1');
        batchNormalizationLayer(Name='input_lyr_b1');
        reluLayer(Name='input_lyr_r1');
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c2');
        batchNormalizationLayer(Name='input_lyr_b2');
        reluLayer(Name='input_lyr_r2');
        convolution2dLayer([3 3],num_feat,Padding="same",Name='input_lyr_c3');
        batchNormalizationLayer(Name='input_lyr_b3');
        reluLayer(Name='input_lyr_r3');
        maxPooling2dLayer([2 2],Stride=[2 2],Name='input_lyr_max');
    ];
    network=dlnetwork(layerinput_b);   


    %% connect coarse stage (p branch)
    % encoder part
    num_cardinal=2;
    radix=2;
    num_layer=2;
    num_feat=[128 256];
    num_block=[2 2];
    % layer 1: (76*76), layer 2: (76*76)
    for i_layer=1:num_layer
        % grab out current block number for current layer
        current_num_block=num_block(i_layer);
        current_num_feat=num_feat(i_layer);
        % if this is the first layer
        for i_block=1:current_num_block
            % initiate bottleneck block
            if i_layer==num_layer
                [network]=dilated_bottleneck(network, i_layer, i_block, current_num_feat);
            else
                [network]=res_block_bottleneck(network, i_layer, i_block, current_num_feat, radix, num_cardinal);
            end
            % connect the input
            if i_layer==1
                if i_block==1
                    network=connectLayers(network, 'input_lyr_max', strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                else
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block-1)), strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                end
            elseif i_layer==num_layer
                if i_block==1
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))),strcat("dilated_b1c1_",string(i_layer),"_",string(i_block)));
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))),strcat("dilated_b2c1_",string(i_layer),"_",string(i_block)));
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))),strcat("dilated_b3c1_",string(i_layer),"_",string(i_block)));
                else
                    network=connectLayers(network, strcat("dilated_fuse_a1_",string(i_layer),"_",string(i_block-1)),strcat("dilated_b1c1_",string(i_layer),"_",string(i_block)));
                    network=connectLayers(network, strcat("dilated_fuse_a1_",string(i_layer),"_",string(i_block-1)),strcat("dilated_b2c1_",string(i_layer),"_",string(i_block)));
                    network=connectLayers(network, strcat("dilated_fuse_a1_",string(i_layer),"_",string(i_block-1)),strcat("dilated_b3c1_",string(i_layer),"_",string(i_block)));
                end
            else
                if i_block==1
                    network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),'_',string(num_block(i_layer-1))), strcat('main_layer_c1_',string(i_layer),'_',string(i_block)));
                else
                    network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block-1)), strcat('main_layer_c1_',string(i_layer),"_",string(i_block)));
                end
            end
            if i_block==current_num_block && i_layer~=2 % insert max pooling to all layers (except for layer 5 (bottleneck))
                current_lyr_maxPool=[
                    maxPooling2dLayer([2 2],Stride=[2 2],Name=strcat('current_lyr_max_',string(i_layer),'_',string(i_block)));
                    ];
                [network]=addLayers(network,current_lyr_maxPool);
                network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(i_block)),strcat('current_lyr_max_',string(i_layer),'_',string(i_block)));
            end
        end

    end


    %% connect coarse stage (c branch)
    de_num_layer=1;
    % decoder part
    for i_layer=(de_num_layer):(-1):1
         current_num_feat=num_feat(i_layer);
         current_num_block=num_block(i_layer);
         % create transposed and concatenational layer
         res_dblock_transcat=[
             transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_transcat_t_',string(i_layer)));
             res_block_pad_cat_layer(strcat('res_dblock_transcat_cat_',string(i_layer)),2);
             res_dblock_spatial_dropout_layer(strcat('res_dblock_drop_',string(i_layer)),0.1);
         ];
         % skip dilated convolution
         [network]=addLayers(network,res_dblock_transcat);
         [network]=skip_attentiongate_bottleneck(network, i_layer, num_feat(i_layer+1));
         [network]=skip_dilated_bottleneck(network, i_layer, current_num_feat);
         % connect the encoder to decoder part
         network=connectLayers(network, strcat('res_block_final_add_',string(i_layer),"_",string(num_block(i_layer))),strcat('skip_attentiongate_en_c1_',string(i_layer)));
         if i_layer==num_layer-1
            network=connectLayers(network, strcat("dilated_fuse_a1_",string(i_layer+1),"_",string(num_block(i_layer+1))),strcat('skip_attentiongate_de_c1_',string(i_layer)));
         else
            network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer+1), "_", string(num_block(i_layer+1))),strcat('skip_attentiongate_de_c1_',string(i_layer)));
         end
         network=connectLayers(network, strcat('skip_attentiongate_mult_m1_',string(i_layer)),strcat("skip_dilated_b1c1_",string(i_layer)));
         network=connectLayers(network, strcat('skip_attentiongate_mult_m1_',string(i_layer)),strcat("skip_dilated_b2c1_",string(i_layer)));
         network=connectLayers(network, strcat('skip_attentiongate_mult_m1_',string(i_layer)),strcat("skip_dilated_b3c1_",string(i_layer)));
         network=connectLayers(network, strcat('skip_attentiongate_mult_m1_',string(i_layer)),strcat("skip_cat_",string(i_layer),'/in1'));
         % connect the bottleneck to transpose convolution
         network=connectLayers(network, strcat("dilated_fuse_a1_",string(i_layer+1),"_",string(num_block(i_layer+1))),strcat('res_dblock_transcat_t_',string(i_layer)));
         % concatenate skip to transpose concatenation
         network=connectLayers(network, strcat("skip_a1_",string(i_layer)),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
         for i_block=1:current_num_block
            [network]=res_dblock_bottleneck_d(network, i_layer, i_block, current_num_feat, radix, num_cardinal);
            if i_block==1
                network=connectLayers(network, strcat('res_dblock_drop_',string(i_layer)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(i_block)));
                network=connectLayers(network, strcat('res_dblock_drop_',string(i_layer)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(i_block)));
            else
                network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(i_block-1)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(i_block)));
                network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(i_block-1)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(i_block)));
            end
         end
    end

    % add final decoder block
    % transpose and concatenation block
    res_dblock_transcat=[
         transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name='res_dblock_d_transcat_t_final');
         res_block_pad_cat_layer('res_dblock_d_transcat_cat_final',2);
    ];
    [network]=addLayers(network,res_dblock_transcat);
    [network]=skip_dilated_bottleneck(network, 0, current_num_feat);
    network=connectLayers(network, 'input_lyr_r3',strcat("skip_dilated_b1c1_",string(0)));
    network=connectLayers(network, 'input_lyr_r3',strcat("skip_dilated_b2c1_",string(0)));
    network=connectLayers(network, 'input_lyr_r3',strcat("skip_dilated_b3c1_",string(0)));
    network=connectLayers(network, 'input_lyr_r3',strcat("skip_cat_",string(0),'/in1'));
    network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(num_block(i_layer))),'res_dblock_d_transcat_t_final');
    network=connectLayers(network, strcat("skip_a1_",string(0)),strcat('res_dblock_d_transcat_cat_final',"/in2"));

    % decoder block
    num_feat=64;
    final_decoder=[
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='final_decoder_d_c1');
        batchNormalizationLayer(Name='final_decoder_d_b1');
        reluLayer(Name='final_decoder_d_r1'); % feature map size 304*304*64
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='final_decoder_d_c2');
        batchNormalizationLayer(Name='final_decoder_d_b2');
        reluLayer(Name='final_decoder_d_r2'); % feature map size 304*304*64
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='final_decoder_d_c3');
        batchNormalizationLayer(Name='final_decoder_d_b3');
        reluLayer(Name='final_decoder_d_r3'); % feature map size 304*304*64
        ];
    network=addLayers(network,final_decoder);
    network=connectLayers(network, 'res_dblock_d_transcat_cat_final','final_decoder_d_c1');

    layer_flatten_coarse=[
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_c_c1');
          sigmoidLayer(Name='layer_flatten_coarse_c_a1');
    ];
    
    network=addLayers(network,layer_flatten_coarse);
    network=connectLayers(network, 'final_decoder_d_r3','layer_flatten_coarse_c_c1');

end