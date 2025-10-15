function [dlnetwork]=skip_attentiongate_bottleneck(dlnetwork, layer_ind, de_num_feat)

    
    %% encoder input branch convolution
    en_input_branch=[
       convolution2dLayer([1 1],de_num_feat,Stride=1,Padding=[0 0],Name=strcat('skip_attentiongate_en_c1_',string(layer_ind)));
    ];
    dlnetwork=addLayers(dlnetwork,en_input_branch);

    
    %% decoder input branch convolution
    de_input_branch=[
       convolution2dLayer([1 1],de_num_feat,Stride=1,Padding=[0 0],Name=strcat('skip_attentiongate_de_c1_',string(layer_ind)));
       transposedConv2dLayer([2 2],de_num_feat,"Stride",[2 2],Name=strcat('skip_attentiongate_de_t1_',string(layer_ind)));
    ];
    dlnetwork=addLayers(dlnetwork,de_input_branch);


    %% addition part
    add_block=[
       additionLayer(2,Name=strcat('skip_attentiongate_add_add1_',string(layer_ind)));
       reluLayer(Name=strcat('skip_attentiongate_add_a1_',string(layer_ind)));
       convolution2dLayer([1 1],1,Stride=1,Padding=[0 0],Name=strcat('skip_attentiongate_add_c1_',string(layer_ind)));
       sigmoidLayer(Name=strcat('skip_attentiongate_add_a2_',string(layer_ind)));
    ];
    dlnetwork=addLayers(dlnetwork,add_block);
    dlnetwork=connectLayers(dlnetwork, strcat('skip_attentiongate_en_c1_',string(layer_ind)),strcat('skip_attentiongate_add_add1_',string(layer_ind),'/in1'));
    dlnetwork=connectLayers(dlnetwork, strcat('skip_attentiongate_de_t1_',string(layer_ind)),strcat('skip_attentiongate_add_add1_',string(layer_ind),'/in2'));


    %% multiplication part
    mult_block=[
       multiplicationLayer(2,Name=strcat('skip_attentiongate_mult_m1_',string(layer_ind)));
    ];
    dlnetwork=addLayers(dlnetwork,mult_block);
    dlnetwork=connectLayers(dlnetwork, strcat('skip_attentiongate_add_a2_',string(layer_ind)),strcat('skip_attentiongate_mult_m1_',string(layer_ind),'/in1'));
    dlnetwork=connectLayers(dlnetwork, strcat('skip_attentiongate_add_a2_',string(layer_ind)),strcat('skip_attentiongate_mult_m1_',string(layer_ind),'/in2'));




end