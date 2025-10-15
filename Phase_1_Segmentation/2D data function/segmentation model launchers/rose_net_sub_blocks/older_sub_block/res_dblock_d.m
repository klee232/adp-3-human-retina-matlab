

function [dlnetwork_obj]=res_dblock_d(dlnetwork, layer_ind, num_cardinal,num_feat)
    
    %% grab out dlnetwork object
    dlnetwork_obj=dlnetwork;    


    %% create transpose block
    res_dblock_tranpose=[
        concatenationLayer(3,2,Name=strcat('res_dblock_d_transpose_cat_',string(layer_ind)));
        transposedConv2dLayer([2 2],num_feat,"Stride",[2 2],Name=strcat('res_dblock_d_transpose_t_',string(layer_ind))); 
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_tranpose);


    %% setup first split layer
    first_split_block_first=[
        input_spilit_first_res_block_layer(strcat("res_dblock_d_first_branch_",string(layer_ind)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,first_split_block_first);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_transpose_t_',string(layer_ind)), strcat("res_dblock_d_first_branch_",string(layer_ind)));

    first_split_block_second=[
        input_spilit_second_res_block_layer(strcat("res_dblock_d_second_branch_",string(layer_ind)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,first_split_block_second);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_transpose_t_',string(layer_ind)), strcat("res_dblock_d_second_branch_",string(layer_ind)));


    %% create resnest block based on cardinal number
    for i_cardinal=1:num_cardinal

        %% first branch of resnest block
        res_dblock_d_first_branch=[
            input_spilit_first_res_block_layer(strcat("res_dblock_d_first_branch_",string(layer_ind), "_", string(i_cardinal)));
            convolution2dLayer([1 1],(num_feat/(2*num_cardinal)),Padding="same",Name=strcat('res_dblock_d_first_branch_b1m1c1_',string(layer_ind),"_",string(i_cardinal)));
            batchNormalizationLayer(Name=strcat('res_dblock_d_first_branch_b1m1b1_',string(layer_ind), "_", string(i_cardinal)));
            reluLayer(Name=strcat('res_dblock_d_first_branch_b1m1a1_',string(layer_ind), "_",string(i_cardinal)));
            convolution2dLayer([3 3],(num_feat/(1*num_cardinal)),Padding="same",Name=strcat('res_dblock_d_first_branch_b1m1c2_',string(layer_ind), "_",string(i_cardinal)));
            batchNormalizationLayer(Name=strcat('res_dblock_d_first_branch_b1m1b2_',string(layer_ind),"_",string(i_cardinal)));
            reluLayer(Name=strcat('res_dblock_d_first_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)));
        ];
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_first_branch);


        %% second branch of resnest block
        res_dblock_d_second_branch=[
            input_spilit_second_res_block_layer(strcat("res_dblock_d_second_branch_",string(layer_ind), "_",string(i_cardinal)));
            convolution2dLayer([1 1],(num_feat/(2*num_cardinal)),Padding="same",Name=strcat('res_dblock_d_second_branch_b1m1c1_',string(layer_ind),"_",string(i_cardinal)));
            batchNormalizationLayer(Name=strcat('res_dblock_d_second_branch_b1m1b1_',string(layer_ind),"_",string(i_cardinal)));
            reluLayer(Name=strcat('res_dblock_d_second_branch_b1m1a1_',string(layer_ind),"_",string(i_cardinal)));
            convolution2dLayer([3 3],(num_feat/(1*num_cardinal)),Padding="same",Name=strcat('res_dblock_d_second_branch_b1m1c2_',string(layer_ind),"_",string(i_cardinal)));
            batchNormalizationLayer(Name=strcat('res_dblock_d_second_branch_b1m1b2_',string(layer_ind),"_",string(i_cardinal)));
            reluLayer(Name=strcat('res_dblock_d_second_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)));
        ];
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_second_branch);

        if i_cardinal==1
            dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_first_branch_",string(layer_ind)), strcat("res_dblock_d_first_branch_",string(layer_ind), "_", string(i_cardinal)));
            dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_first_branch_",string(layer_ind)), strcat("res_dblock_d_second_branch_",string(layer_ind), "_", string(i_cardinal)));
        else
            dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_second_branch_",string(layer_ind)), strcat("res_dblock_d_first_branch_",string(layer_ind), "_", string(i_cardinal)));
            dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_second_branch_",string(layer_ind)), strcat("res_dblock_d_second_branch_",string(layer_ind), "_", string(i_cardinal)));
        end

        %% split attention additional branch
        res_dblock_d_split_atten_add_branch=[
            additionLayer(2,Name=strcat('res_dblock_d_split_atten_add_',string(layer_ind),"_",string(i_cardinal)));
            res_block_global_average_pooling_layer(strcat("res_dblock_d_split_atten_avg_pool_",string(layer_ind),"_",string(i_cardinal)));  
            fullyConnectedLayer(num_feat/(2*num_cardinal),Name=strcat("res_dblock_d_split_atten_fc_",string(layer_ind),"_",string(i_cardinal)));
            batchNormalizationLayer(Name=strcat("res_dblock_d_split_atten_b_",string(layer_ind),"_",string(i_cardinal)));
            reluLayer(Name=strcat("res_dblock_d_split_atten_a_",string(layer_ind),"_",string(i_cardinal)));
        ];
        res_dblock_d_split_atten_add_branch_1=[
            fullyConnectedLayer((num_feat/(1*num_cardinal)),Name=strcat("res_dblock_d_split_atten_branch_1_fc_",string(layer_ind),"_",string(i_cardinal)));
            softmaxLayer(Name=strcat("res_dblock_d_split_atten_branch_1_softmax_",string(layer_ind),"_",string(i_cardinal)));
        ];
        res_dblock_d_split_atten_add_branch_2=[
            fullyConnectedLayer((num_feat/(1*num_cardinal)),Name=strcat("res_dblock_d_split_atten_branch_2_fc_",string(layer_ind),"_",string(i_cardinal)));
            softmaxLayer(Name=strcat("res_dblock_d_split_atten_branch_2_softmax_",string(layer_ind),"_",string(i_cardinal)));
        ];
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_add_branch);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_first_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_add_',string(layer_ind),"_",string(i_cardinal),'/in1'));
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_second_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_add_',string(layer_ind),"_",string(i_cardinal),'/in2'));
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_add_branch_1);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_split_atten_a_",string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_branch_1_fc_',string(layer_ind),"_",string(i_cardinal)));
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_add_branch_2);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_d_split_atten_a_",string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_branch_2_fc_',string(layer_ind),"_",string(i_cardinal)));
    
    
        %% split attention multiplication branch
        res_dblock_d_split_atten_mult_branch_1=[
            res_block_chn_multiplication_layer(2,strcat("res_dblock_d_branch_1_chn_multiplication_layer_",string(layer_ind),"_",string(i_cardinal)));  
        ];
        res_dblock_d_split_atten_mult_branch_2=[
            res_block_chn_multiplication_layer(2,strcat("res_dblock_d_branch_2_chn_multiplication_layer_",string(layer_ind),"_",string(i_cardinal)));  
        ];
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_mult_branch_1);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_split_atten_branch_1_softmax_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_branch_1_chn_multiplication_layer_',string(layer_ind),"_",string(i_cardinal),"/in2"));
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_first_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_branch_1_chn_multiplication_layer_',string(layer_ind),'_', string(i_cardinal),"/in1"));
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_mult_branch_2);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_split_atten_branch_2_softmax_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_branch_2_chn_multiplication_layer_',string(layer_ind),"_",string(i_cardinal),"/in2"));
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_second_branch_b1m1a2_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_branch_2_chn_multiplication_layer_',string(layer_ind),"_", string(i_cardinal),"/in1"));

        % additional layer
        res_dblock_d_split_atten_branch=[
           additionLayer(2,Name=strcat('res_dblock_d_split_atten_branch_add_',string(layer_ind),"_",string(i_cardinal)));
        ];
        dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_split_atten_branch);
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_branch_1_chn_multiplication_layer_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_branch_add_',string(layer_ind),"_",string(i_cardinal),'/in1'));
        dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_branch_2_chn_multiplication_layer_',string(layer_ind),"_",string(i_cardinal)), strcat('res_dblock_d_split_atten_branch_add_',string(layer_ind),"_",string(i_cardinal),'/in2'));
  
    end


    %% concatenate the fianl part of res block
    res_dblock_d_final_cat=[
        concatenationLayer(3,num_cardinal,Name=strcat('res_dblock_d_final_cat_',string(layer_ind)));
        convolution2dLayer([1 1],num_feat,Name=strcat('res_dblock_d_final_cat_c1_',string(layer_ind)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_final_cat);
    for i_cardinal=1:num_cardinal     
        dlnetwork_obj=connectLayers(dlnetwork_obj,strcat('res_dblock_d_split_atten_branch_add_',string(layer_ind),"_",string(i_cardinal)),strcat('res_dblock_d_final_cat_',string(layer_ind),'/in',string(i_cardinal)));
    end


    %% add the final part of res block
    res_dblock_d_final_add=[
        additionLayer(2,Name=strcat('res_dblock_d_final_add_',string(layer_ind)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_d_final_add);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_final_cat_c1_',string(layer_ind)), strcat('res_dblock_d_final_add_',string(layer_ind),'/in1'));
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_d_transpose_t_',string(layer_ind)), strcat('res_dblock_d_final_add_',string(layer_ind),'/in2'));



end