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


function [network]=rose_model_launcher(input)
    [img_depth,img_row,img_col,~]=size(input);
    
    %% input block 
    layerinput_b=[
        image3dInputLayer([img_depth img_row img_col 1],Normalization="none",Name="input_lyr");
    ];
    network=dlnetwork(layerinput_b);   

    % input SVC split block
    layerinput_s_s=[
        input_spilit_SVC_layer("input_spilit_s_lyr"); 
    ];
    network=addLayers(network,layerinput_s_s);
    network=connectLayers(network, 'input_lyr', 'input_spilit_s_lyr');


    %% connect coarse stage (p branch)
    % encoder part
    num_cardinal=2;
    num_layer=5;
    num_feat=[64  128 256 512 1024];
    for i_layer=1:num_layer

        % if this is the last layer (connect the bottleneck instead of
        % original renest block)
        if i_layer==5
            [network]=res_bottleneck_block(network, i_layer, num_cardinal,num_feat(i_layer));
        else
            [network]=res_block(network, i_layer, num_cardinal,num_feat(i_layer));
        end

        % if this is the first layer, connect the block with the input
        if i_layer==1
            for i_cardinal=1:num_cardinal
                network=connectLayers(network, 'input_spilit_s_lyr', strcat('res_block_first_branch_',string(i_layer),"_",string(i_cardinal)));
                network=connectLayers(network, 'input_spilit_s_lyr', strcat('res_block_second_branch_',string(i_layer),"_",string(i_cardinal)));
            end
            network=connectLayers(network, 'input_spilit_s_lyr', strcat('res_block_final_add_',string(i_layer),'/in2'));
        % otherwise, connect to the output of the previous layer
        else
            for i_cardinal=1:num_cardinal
                network=connectLayers(network, strcat('res_block_final_max_',string(i_layer-1)), strcat('res_block_first_branch_',string(i_layer),"_",string(i_cardinal)));
                network=connectLayers(network, strcat('res_block_final_max_',string(i_layer-1)), strcat('res_block_second_branch_',string(i_layer),"_",string(i_cardinal)));
            end
            network=connectLayers(network, strcat('res_block_final_max_',string(i_layer-1)), strcat('res_block_final_add_',string(i_layer),'/in2'));
        end
    end


    % decoder part
    for i_layer=(num_layer-1):(-1):1
         [network]=res_dblock(network, i_layer, num_cardinal,num_feat(i_layer));
         % if the current layer is 4, connect the bottleneck and encoder layer 4 to the concatenation layer
         if i_layer==4
            network=connectLayers(network, strcat('res_block_final_add_',string(5)), strcat('res_dblock_transpose_cat_',string(i_layer),'/in1'));
            network=connectLayers(network, strcat('res_block_final_max_',string(i_layer)), strcat('res_dblock_transpose_cat_',string(i_layer),'/in2'));
         else
            network=connectLayers(network, strcat('res_dblock_final_add_',string(i_layer+1)), strcat('res_dblock_transpose_cat_',string(i_layer),'/in1'));
            network=connectLayers(network, strcat('res_block_final_max_',string(i_layer)), strcat('res_dblock_transpose_cat_',string(i_layer),'/in2'));
         end
    end


    %% connect coarse stage (c branch)
    % decoder part
    for i_layer=(num_layer-3):(-1):1
         [network]=res_dblock_d(network, i_layer, num_cardinal,num_feat(i_layer));
         % if the current layer is 4, connect the bottleneck and encoder layer 4 to the concatenation layer
         if i_layer==2
            network=connectLayers(network, strcat('res_block_final_add_',string(3)), strcat('res_dblock_d_transpose_cat_',string(i_layer),'/in1'));
            network=connectLayers(network, strcat('res_block_final_max_',string(i_layer)), strcat('res_dblock_d_transpose_cat_',string(i_layer),'/in2'));
         else
            network=connectLayers(network, strcat('res_dblock_d_final_add_',string(i_layer+1)), strcat('res_dblock_d_transpose_cat_',string(i_layer),'/in1'));
            network=connectLayers(network, strcat('res_block_final_max_',string(i_layer)), strcat('res_dblock_d_transpose_cat_',string(i_layer),'/in2'));
         end
    end


    %% connect fine stage (SRS module)
    filter_size=3;
    num_feat=filter_size*filter_size;
    fine_block=[
        concatenationLayer(4,3,Name='fine_block_cat');
        convolution3dLayer([1 3 3],num_feat,Padding="same",Name='fine_block_c1');
        batchNormalizationLayer(Name='fine_block_b1');
        reluLayer(Name='fine_block_r1');
        convolution3dLayer([1 3 3],num_feat,Padding="same",Name='fine_block_c2');
        batchNormalizationLayer(Name='fine_block_b2');
        reluLayer(Name='fine_block_r2');
        convolution3dLayer([1 3 3],num_feat,Padding="same",Name='fine_block_c3');
        batchNormalizationLayer(Name='fine_block_b3');
        reluLayer(Name='fine_block_r3');
    ];
    network=addLayers(network,fine_block);
    network=connectLayers(network, 'input_spilit_s_lyr','fine_block_cat/in1');
    network=connectLayers(network, 'res_dblock_final_add_1','fine_block_cat/in2');
    network=connectLayers(network, 'res_dblock_d_final_add_1','fine_block_cat/in3');


    %% connect fine stage (propagation coefficients stage)
    fine_block_p=[
        concatenationLayer(4,2,Name='fine_block_p_cat');
        fine_block_propagation_layer('fine_block_p',filter_size);
    ];
    network=addLayers(network,fine_block_p);
    network=connectLayers(network, 'res_dblock_final_add_1', 'fine_block_p_cat/in1');
    network=connectLayers(network, 'fine_block_r3','fine_block_p_cat/in2');

    fine_block_c=[
        concatenationLayer(4,2,Name='fine_block_c_cat');
        fine_block_propagation_layer('fine_block_c',filter_size);
    ];
    network=addLayers(network,fine_block_c);
    network=connectLayers(network, 'res_dblock_d_final_add_1', 'fine_block_c_cat/in1');
    network=connectLayers(network, 'fine_block_r2','fine_block_c_cat/in2');

    fine_block_mult=[
        multiplicationLayer(2,Name='fine_block_mult');
    ];
    network=addLayers(network,fine_block_mult);
    network=connectLayers(network, 'fine_block_p', 'fine_block_mult/in1');
    network=connectLayers(network, 'fine_block_c', 'fine_block_mult/in2');



end