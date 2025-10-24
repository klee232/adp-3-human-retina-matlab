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


function [network]=rose_1_model_SVC_launcher(input)
    [img_row,img_col,~]=size(input);
   
    
    %% setup input layer (conv1 in the original model) (checked)
    num_feat=64;
    layerinput_b=[
        imageInputLayer([img_row img_col 1],Normalization="none",Name="input_lyr");
        convolution2dLayer([3 3],num_feat,Stride=2,Padding=[1 1],Name='input_lyr_c1');
        batchNormalizationLayer(Name='input_lyr_b1');
        reluLayer(Name='input_lyr_r1');
        convolution2dLayer([3 3],num_feat,Stride=1,Padding=[1 1],Name='input_lyr_c2');
        batchNormalizationLayer(Name='input_lyr_b2');
        reluLayer(Name='input_lyr_r2');
        convolution2dLayer([3 3],num_feat*2,Stride=1,Padding=[1 1],Name='input_lyr_c3');
        batchNormalizationLayer(Name='input_lyr_b3');
        reluLayer(Name='input_lyr_r3');
        maxPooling2dLayer([3 3],Stride=[2 2], Padding=[1 1],Name='input_lyr_max');
        ];
    network=dlnetwork(layerinput_b); % this output 76*76


    %% connect coarse stage (p branch)
    % encoder part
    num_cardinal=2;
    radix=2;
    num_layer=4;
    num_feat=[64 128 256 512];
    num_block=[3 4 6 3];
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
            if i_block==current_num_block && i_block~=1
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
    num_layer=5;
    num_feat=[64 128 256 512 1024];
    for i_layer=num_layer:(-1):1
        current_num_feat=num_feat(i_layer);
              
        % if this is the shallowest layer, don't do any feature concatenation
        if i_layer==1 % (checked)
            res_dblock_transcat=[
                transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_transcat_t_',string(i_layer)));
            ];
            [network]=addLayers(network,res_dblock_transcat);     
            network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer+1), "_", string(1)),strcat('res_dblock_transcat_t_',string(i_layer)));
            [network]=res_dblock_bottleneck(network, i_layer, 1, current_num_feat, radix, num_cardinal);
            network=connectLayers(network, strcat('res_dblock_transcat_t_',string(i_layer)), strcat("d_main_layer_c1_",string(i_layer),"_",string(1)));
            network=connectLayers(network, strcat('res_dblock_transcat_t_',string(i_layer)), strcat('d_residual_layer_c1_',string(i_layer), "_", string(1)));

        % otherwise, conduct feature concatenation
        else
            % create transposed and concatenation block if this is the
            % first block
            res_dblock_transcat=[
                transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_transcat_t_',string(i_layer)));
                res_block_pad_cat_layer(strcat('res_dblock_transcat_cat_',string(i_layer)),2);
            ];
            [network]=addLayers(network,res_dblock_transcat);
            if i_layer==5 % checked
                network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-1),"_",string(num_block(i_layer-1))),strcat('res_dblock_transcat_t_',string(i_layer)));
                network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-2),"_",string(num_block(i_layer-2))),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
            elseif i_layer==3
                network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer+1), "_", string(1)),strcat('res_dblock_transcat_t_',string(i_layer)));
                network=connectLayers(network, strcat('res_block_final_add_',string(i_layer-2),"_",string(num_block(i_layer-2))),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
            elseif i_layer==2 % checked
                network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer+1), "_", string(1)),strcat('res_dblock_transcat_t_',string(i_layer)));
                network=connectLayers(network, 'input_lyr_r3',strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
            else % checked
                network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer+1), "_", string(1)),strcat('res_dblock_transcat_t_',string(i_layer)));
                network=connectLayers(network, strcat('current_lyr_max_',string(i_layer-2),"_",string(num_block(i_layer-2))),strcat('res_dblock_transcat_cat_',string(i_layer),"/in2"));
            end
            % initiate bottleneck block
            [network]=res_dblock_bottleneck(network, i_layer, 1, current_num_feat, radix, num_cardinal);
            network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)),strcat('d_main_layer_c1_',string(i_layer),'_',string(1)));
            network=connectLayers(network, strcat('res_dblock_transcat_cat_',string(i_layer)), strcat('d_residual_layer_c1_',string(i_layer), "_", string(1)));
        end

    end

    layer_flatten_coarse_p=[
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_p_c1');
          sigmoidLayer(Name='layer_flatten_coarse_p_a1');
    ];
    network=addLayers(network,layer_flatten_coarse_p);
    network=connectLayers(network, strcat('d_drop_layer_a1_',string(i_layer), "_", string(1)),'layer_flatten_coarse_p_c1');


    %% connect coarse stage (c branch)
    % decoder part
    for i_layer=(num_layer-3):(-1):1     
         current_num_feat=num_feat(i_layer);
         % if the current layer is 4, connect the bottleneck and encoder layer 4 to the concatenation layer
         if i_layer==2
             res_dblock_d_transcat=[
                 transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_d_transcat_t_',string(i_layer)));
                 res_block_pad_cat_layer(strcat('res_dblock_d_transcat_cat_',string(i_layer)),2);
             ];
             [network]=addLayers(network,res_dblock_d_transcat);
             network=connectLayers(network, strcat('res_block_final_add_',string(i_layer-1),"_",string(num_block(i_layer-1))),strcat('res_dblock_d_transcat_t_',string(i_layer)));
             network=connectLayers(network, 'input_lyr_r3',strcat('res_dblock_d_transcat_cat_',string(i_layer),"/in2"));
             [network]=res_dblock_bottleneck_d(network, i_layer, 1, current_num_feat, radix, num_cardinal);
             network=connectLayers(network, strcat('res_dblock_d_transcat_cat_',string(i_layer)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(1)));
             network=connectLayers(network, strcat('res_dblock_d_transcat_cat_',string(i_layer)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(1)));
         else
            res_dblock_d_transcat=[
                 transposedConv2dLayer([2 2],current_num_feat,"Stride",[2 2],Name=strcat('res_dblock_d_transcat_t_',string(i_layer)));
             ];
            [network]=addLayers(network,res_dblock_d_transcat);
            network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer+1), "_", string(1)),strcat('res_dblock_d_transcat_t_',string(i_layer)));
            [network]=res_dblock_bottleneck_d(network, i_layer, 1, current_num_feat, radix, num_cardinal);
            network=connectLayers(network, strcat('res_dblock_d_transcat_t_',string(i_layer)), strcat("d_main_layer_d_c1_",string(i_layer),"_",string(1)));
            network=connectLayers(network, strcat('res_dblock_d_transcat_t_',string(i_layer)), strcat('d_residual_d_layer_c1_',string(i_layer), "_", string(1)));
         end
         
    end

    layer_flatten_coarse_c=[
          convolution2dLayer([1 1],1,Padding="same",Name='layer_flatten_coarse_c_c1');
          sigmoidLayer(Name='layer_flatten_coarse_c_a1');
    ];
    network=addLayers(network,layer_flatten_coarse_c);
    network=connectLayers(network, strcat('d_drop_layer_d_a1_',string(i_layer), "_", string(1)),'layer_flatten_coarse_c_c1');


    %% connect fine stage (SRS module) (p-branch)
    filter_size=5;
    num_feat=filter_size*filter_size;
    fine_block=[
        concatenationLayer(3,3,Name='fine_block_cat');
        convolution2dLayer([3 3],256,Padding="same",Name='fine_block_c1',WeightsInitializer="narrow-normal");
        batchNormalizationLayer(Name='fine_block_b1');
        reluLayer(Name='fine_block_r1');
        convolution2dLayer([3 3],512,Padding="same",Name='fine_block_c2',WeightsInitializer="narrow-normal");
        batchNormalizationLayer(Name='fine_block_b2');
        reluLayer(Name='fine_block_r2');
        convolution2dLayer([3 3],num_feat,Padding="same",Name='fine_block_p_c3',WeightsInitializer="narrow-normal", BiasInitializer=@(sz) single(cat(3,zeros(1,1,floor(num_feat/2)), 1, zeros(1,1,(sz(3)-floor(num_feat/2)-1)))));
        batchNormalizationLayer(Name='fine_block_p_b3');
        reluLayer(Name='fine_block_p_r3');
        fine_block_softmax_layer('fine_block_p_a1')
    ];
    network=addLayers(network,fine_block);
    network=connectLayers(network, 'input_lyr','fine_block_cat/in1');
    network=connectLayers(network, 'layer_flatten_coarse_p_a1','fine_block_cat/in2');
    network=connectLayers(network, 'layer_flatten_coarse_c_a1','fine_block_cat/in3');


    %% connect fine stage (SRS module) (c-branch)
    fine_block_c=[
        convolution2dLayer([3 3],num_feat,Padding="same",Name='fine_block_c_c3',WeightsInitializer="narrow-normal", BiasInitializer=@(sz) single(cat(3,zeros(1,1,floor(num_feat/2)), 1, zeros(1,1,(sz(3)-floor(num_feat/2)-1)))));
        batchNormalizationLayer(Name='fine_block_c_b3');
        reluLayer(Name='fine_block_c_r3');
        fine_block_softmax_layer('fine_block_c_a1')
    ];
    network=addLayers(network,fine_block_c);
    network=connectLayers(network, 'fine_block_c2','fine_block_c_c3');


    %% connect fine stage (propagation coefficients stage)
    fine_block_p=[
        fine_block_propagation_layer(3,5,'fine_block_p');
    ];
    network=addLayers(network,fine_block_p);
    network=connectLayers(network, 'layer_flatten_coarse_p_a1','fine_block_p/in1');
    network=connectLayers(network, 'layer_flatten_coarse_c_a1','fine_block_p/in2');
    network=connectLayers(network, 'fine_block_p_a1','fine_block_p/in3');

    fine_block_c=[
        fine_block_propagation_layer(3,5,'fine_block_c');
    ];
    network=addLayers(network,fine_block_c);
    network=connectLayers(network, 'layer_flatten_coarse_p_a1','fine_block_c/in1');
    network=connectLayers(network, 'layer_flatten_coarse_c_a1','fine_block_c/in2');
    network=connectLayers(network, 'fine_block_c_a1','fine_block_c/in3');

    fine_block_fusion=[
        fine_block_fusion_layer(2,'fine_block_fusion');
        sigmoidLayer(Name='fine_block_a1')
    ];
    
    network=addLayers(network,fine_block_fusion);
    network=connectLayers(network, 'fine_block_p', 'fine_block_fusion/in1');
    network=connectLayers(network, 'fine_block_c', 'fine_block_fusion/in2');


end