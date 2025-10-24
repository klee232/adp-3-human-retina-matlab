
% num_feat: intended feature number for current layer
% radix: number of partition for this layer
% num_cardinal: number of cardinal for this layer


function [dlnetwork_obj]=res_dblock_bottleneck(dlnetwork, layer_ind, i_block, num_feat, radix, num_cardinal)
    
    %% grab out dlnetwork object
    dlnetwork_obj=dlnetwork;


    %% setup first main layer (checked) original conv1
    % this part enlarge the input feature map to the assigned number of
    % feature maps
    d_main_layer=[
        convolution2dLayer([3 3],num_feat,Padding="same",Name=strcat('d_main_layer_c1_',string(layer_ind), "_", string(i_block)));
        batchNormalizationLayer(Name=strcat('d_main_layer_b1_',string(layer_ind), "_", string(i_block)));
        reluLayer(Name=strcat('d_main_layer_a1_',string(layer_ind), "_", string(i_block)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,d_main_layer);
    

    %% create resnest block spatial split attention part
    %% resnet block convolutional part
    num_group=num_cardinal*radix;
    res_dblock_conv=[
        groupedConvolution2dLayer([3 3],((num_feat)*radix/num_group),num_group,Padding="same",Name=strcat('res_dblock_b1m1c1_',string(layer_ind),"_",string(i_block)));
        batchNormalizationLayer(Name=strcat('res_dblock_b1m1b1_',string(layer_ind), "_", string(i_block)));
        reluLayer(Name=strcat('res_dblock_b1m1a1_',string(layer_ind), "_", string(i_block)));
        ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_conv);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('d_main_layer_a1_',string(layer_ind),"_",string(i_block)), strcat('res_dblock_b1m1c1_',string(layer_ind),"_",string(i_block)));


    %% resnet block split attention additional part
    res_dblock_split_atten_add=[
        res_block_input_addition_layer(strcat('res_dblock_split_atten_add_',string(layer_ind), "_", string(i_block)), radix, num_cardinal);
        res_block_global_average_pooling_layer(strcat("res_dblock_split_atten_avg_pool_",string(layer_ind), "_", string(i_block)));        
        groupedConvolution2dLayer([1 1],((num_feat)*radix/num_cardinal),num_cardinal,Padding="same",Name=strcat('res_dblock_split_atten_c1_',string(layer_ind),"_",string(i_block)));
        batchNormalizationLayer(Name=strcat("res_dblock_split_atten_b_",string(layer_ind), "_", string(i_block)));
        groupedConvolution2dLayer([1 1],((num_feat)*radix/num_cardinal),num_cardinal,Padding="same",Name=strcat('res_dblock_split_atten_c2_',string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_split_atten_add);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_b1m1a1_',string(layer_ind), "_", string(i_block)), strcat('res_dblock_split_atten_add_',string(layer_ind), "_", string(i_block)));


    %% split attention multiplication branch
    res_dblock_split_atten_mult_branch=[
        softmaxLayer(Name=strcat("res_dblock_split_atten_mult_r1_",string(layer_ind),"_",string(i_block)));
        res_block_input_split_layer(strcat("res_dblock_split_atten_mult_split_",string(layer_ind),"_",string(i_block)),radix,num_cardinal);
        res_block_chn_multAndadd_layer(2,strcat("res_dblock_split_atten_mult_multAdd_layer_",string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_split_atten_mult_branch);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_split_atten_c2_',string(layer_ind),"_",string(i_block)), strcat("res_dblock_split_atten_mult_r1_",string(layer_ind),"_",string(i_block)));
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_b1m1a1_',string(layer_ind), "_", string(i_block)), strcat("res_dblock_split_atten_mult_multAdd_layer_",string(layer_ind),"_",string(i_block),"/in2"));


    %% concatenate the fianl part of res block
    res_dblock_final_cat=[
        res_block_splat_conatenate_layer(strcat('res_dblock_final_cat_',string(layer_ind),'_',string(i_block)));
        convolution2dLayer([1 1],num_feat,Name=strcat('res_dblock_final_cat_c1_',string(layer_ind),"_",string(i_block)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_final_cat);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat("res_dblock_split_atten_mult_multAdd_layer_",string(layer_ind),"_",string(i_block)), strcat('res_dblock_final_cat_',string(layer_ind),"_",string(i_block)));

    
    %% direct residual part
    d_residual_layer=[
        convolution2dLayer([1 1],num_feat,Padding="same",Name=strcat('d_residual_layer_c1_',string(layer_ind), "_", string(i_block)));
        batchNormalizationLayer(Name=strcat('d_residual_layer_b1_',string(layer_ind), "_", string(i_block)));
        reluLayer(Name=strcat('d_residual_layer_a1_',string(layer_ind), "_", string(i_block)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,d_residual_layer);


    %% add the final part of res block
    res_dblock_final_add=[
        additionLayer(2,Name=strcat('res_dblock_final_add_',string(layer_ind),"_",string(i_block)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,res_dblock_final_add);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_final_cat_c1_',string(layer_ind),"_",string(i_block)), strcat('res_dblock_final_add_',string(layer_ind),"_",string(i_block),'/in1'));
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('d_residual_layer_a1_',string(layer_ind),'_',string(i_block)), strcat('res_dblock_final_add_',string(layer_ind),'_',string(i_block),'/in2'));

    
    %% add downsampling part of res block
    drop_block=[
        convolution2dLayer([1 1],num_feat,Padding="same",Name=strcat('d_drop_layer_c1_',string(layer_ind), "_", string(i_block)));
        batchNormalizationLayer(Name=strcat('d_drop_layer_b1_',string(layer_ind), "_", string(i_block)));
        reluLayer(Name=strcat('d_drop_layer_a1_',string(layer_ind), "_", string(i_block)));
    ];
    dlnetwork_obj=addLayers(dlnetwork_obj,drop_block);
    dlnetwork_obj=connectLayers(dlnetwork_obj, strcat('res_dblock_final_add_',string(layer_ind),"_",string(i_block)), strcat('d_drop_layer_c1_',string(layer_ind), "_", string(i_block)));

end