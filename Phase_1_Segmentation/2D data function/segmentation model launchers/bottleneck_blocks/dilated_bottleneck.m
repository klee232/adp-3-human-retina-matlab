function [dlnetwork]=dilated_bottleneck(dlnetwork, layer_ind, i_block, num_feat)


    %% dilated convolution module
    %% branch original image
    dil_conv_br1=[
        convolution2dLayer([3 3],num_feat,"WeightsInitializer",'he','DilationFactor',[1 1],Name=strcat("dilated_b1c1_",string(layer_ind),"_",string(i_block)),Padding="same");
        batchNormalizationLayer(Name=strcat("dilated_b1b1_",string(layer_ind),"_",string(i_block)));
        reluLayer(Name=strcat("dilated_b1a1_",string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork=addLayers(dlnetwork,dil_conv_br1);
    % branch 2
    dil_conv_br2=[
        convolution2dLayer([3 3],num_feat,"WeightsInitializer",'he','DilationFactor',[4 4],Name=strcat("dilated_b2c1_",string(layer_ind),"_",string(i_block)),Padding="same");
        batchNormalizationLayer(Name=strcat("dilated_b2b1_",string(layer_ind),"_",string(i_block)));
        reluLayer(Name=strcat("dilated_b2a1_",string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork=addLayers(dlnetwork,dil_conv_br2);
    % branch 3
    dil_conv_org_br3=[
        convolution2dLayer([3 3],num_feat,"WeightsInitializer",'he','DilationFactor',[8 8],Name=strcat("dilated_b3c1_",string(layer_ind),"_",string(i_block)),Padding="same");
        batchNormalizationLayer(Name=strcat("dilated_b3b1_",string(layer_ind),"_",string(i_block)));
        reluLayer(Name=strcat("dilated_b3a1_",string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork=addLayers(dlnetwork,dil_conv_org_br3);
    % concatenation
    dil_cat=[
         concatenationLayer(3,3,"Name",strcat("dilated_cat_",string(layer_ind),"_",string(i_block)));
    ];
    dlnetwork=addLayers(dlnetwork,dil_cat);
    dlnetwork=connectLayers(dlnetwork, strcat("dilated_b1a1_",string(layer_ind),"_",string(i_block)),strcat("dilated_cat_",string(layer_ind),"_",string(i_block),'/in1'));
    dlnetwork=connectLayers(dlnetwork, strcat("dilated_b2a1_",string(layer_ind),"_",string(i_block)),strcat("dilated_cat_",string(layer_ind),"_",string(i_block),'/in2'));
    dlnetwork=connectLayers(dlnetwork, strcat("dilated_b3a1_",string(layer_ind),"_",string(i_block)),strcat("dilated_cat_",string(layer_ind),"_",string(i_block),'/in3'));
    % fusion
    dil_conv_org_fuse=[
        convolution2dLayer([1 1],num_feat,"WeightsInitializer",'he',Name=strcat("dilated_fuse_c1_",string(layer_ind),"_",string(i_block)),Padding="same");
        batchNormalizationLayer(Name=strcat("dilated_fuse_b1_",string(layer_ind),"_",string(i_block)));
        reluLayer(Name=strcat("dilated_fuse_a1_",string(layer_ind),"_",string(i_block)));
        ];
    dlnetwork=addLayers(dlnetwork,dil_conv_org_fuse);
    dlnetwork=connectLayers(dlnetwork, strcat("dilated_cat_",string(layer_ind),"_",string(i_block)),strcat("dilated_fuse_c1_",string(layer_ind),"_",string(i_block)));


   
end