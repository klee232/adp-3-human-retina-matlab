function network=fine_stage_prototype_DVC(input)

    [img_row,img_col,~]=size(input);

    %% connect input layers
    input_layer=[
      imageInputLayer([img_row img_col 2],Normalization="none",Name="input_lyr_1");
    ];
    network=dlnetwork(input_layer);


    %% dilated convolution module
    %% branch original image
    input_org_split=[
        fine_block_input_split_layer("org_input",1);
    ];
    network=addLayers(network,input_org_split);
    network=connectLayers(network, 'input_lyr_1','org_input'); 
    dil_conv_org_br1=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[1 1],Name="org_b1_c1",Padding="same");
        batchNormalizationLayer(Name="org_b1_b1");
        reluLayer(Name="org_b1_a1");
        ];
    network=addLayers(network,dil_conv_org_br1);
    network=connectLayers(network, 'org_input','org_b1_c1'); 
    % branch 2
    dil_conv_org_br2=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[2 2],Name="org_b2_c1",Padding="same");
        batchNormalizationLayer(Name="org_b2_b1");
        reluLayer(Name="org_b2_a1");
        ];
    network=addLayers(network,dil_conv_org_br2);
    network=connectLayers(network, 'org_input','org_b2_c1');
    % branch 3
    dil_conv_org_br3=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[4 4],Name="org_b3_c1",Padding="same");
        batchNormalizationLayer(Name="org_b3_b1");
        reluLayer(Name="org_b3_a1");
        ];
    network=addLayers(network,dil_conv_org_br3);
    network=connectLayers(network, 'org_input','org_b3_c1');
    % concatenation
    cat_org=[
         concatenationLayer(3,3,"Name","org_cat");
    ];
    network=addLayers(network,cat_org);
    network=connectLayers(network, 'org_b1_a1','org_cat/in1');
    network=connectLayers(network, 'org_b2_a1','org_cat/in2');
    network=connectLayers(network, 'org_b3_a1','org_cat/in3');
    % fusion
    conv_fuse_org=[
        convolution2dLayer([1 1],64,"WeightsInitializer",'he',Name="fuse_org_c1",Padding="same");
        batchNormalizationLayer(Name="fuse_org_b1");
        reluLayer(Name="fuse_org_a1");
        ];
    network=addLayers(network,conv_fuse_org);
    network=connectLayers(network, 'org_cat','fuse_org_c1');


    %% branch thin
    input_thn_split=[
        fine_block_input_split_layer("thn_input",2);
    ];
    network=addLayers(network,input_thn_split);
    network=connectLayers(network, 'input_lyr_1','thn_input'); 
    dil_conv_thn_br1=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[1 1],Name="thn_b1_c1",Padding="same");
        batchNormalizationLayer(Name="thn_b1_b1");
        reluLayer(Name="thn_b1_a1");
        ];
    network=addLayers(network,dil_conv_thn_br1);
    network=connectLayers(network, 'thn_input','thn_b1_c1'); 
    % branch 2
    dil_conv_thn_br2=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[2 2],Name="thn_b2_c1",Padding="same");
        batchNormalizationLayer(Name="thn_b2_b1");
        reluLayer(Name="thn_b2_a1");
        ];
    network=addLayers(network,dil_conv_thn_br2);
    network=connectLayers(network, 'thn_input','thn_b2_c1');
    % branch 3
    dil_conv_thn_br3=[
        convolution2dLayer([3 3],256,"WeightsInitializer",'he','DilationFactor',[3 3],Name="thn_b3_c1",Padding="same");
        batchNormalizationLayer(Name="thn_b3_b1");
        reluLayer(Name="thn_b3_a1");
        ];
    network=addLayers(network,dil_conv_thn_br3);
    network=connectLayers(network, 'thn_input','thn_b3_c1');
    % concatenation
    cat_thn=[
         concatenationLayer(3,3,"Name","thn_cat");
    ];
    network=addLayers(network,cat_thn);
    network=connectLayers(network, 'thn_b1_a1','thn_cat/in1');
    network=connectLayers(network, 'thn_b2_a1','thn_cat/in2');
    network=connectLayers(network, 'thn_b3_a1','thn_cat/in3');
    % fusion
    conv_fuse_thn=[
        convolution2dLayer([1 1],64,"WeightsInitializer",'he',Name="fuse_thn_c1",Padding="same");
        batchNormalizationLayer(Name="fuse_thn_b1");
        reluLayer(Name="fuse_thn_a1");
        ];
    network=addLayers(network,conv_fuse_thn);
    network=connectLayers(network, 'thn_cat','fuse_thn_c1');


   %% fuse three branches
   % concatenation
   cat_all=[
         concatenationLayer(3,2,"Name","all_cat");
    ];
    network=addLayers(network,cat_all);
    network=connectLayers(network, 'fuse_org_a1','all_cat/in1');
    network=connectLayers(network, 'fuse_thn_a1','all_cat/in2');
    % fusion
    conv_fuse=[
        convolution2dLayer([1 1],64,"WeightsInitializer",'he',Name="fuse_c1",Padding="same");
        batchNormalizationLayer(Name="fuse_b1");
        reluLayer(Name="fuse_a1");
        convolution2dLayer([1 1],64,"WeightsInitializer",'he',Name="fuse_c2",Padding="same");
        batchNormalizationLayer(Name="fuse_b2");
        reluLayer(Name="fuse_a2");
     ];
    network=addLayers(network,conv_fuse);
    network=connectLayers(network, 'all_cat','fuse_c1');
    % spatial convolution
    spa_conv=[
        convolution2dLayer([3 3],1,"WeightsInitializer",'he',Name="spa_c1",Padding="same");
        % batchNormalizationLayer(Name="spa_b1");
        sigmoidLayer(Name="spa_a1");
    ];
    network=addLayers(network,spa_conv);
    network=connectLayers(network, 'fuse_a2','spa_c1');

 
    % %% residual layer
    % resd_block=[
    %     additionLayer(2,"Name","residual_block");
    % ];
    % network=addLayers(network,resd_block);
    % network=connectLayers(network, 'input_lyr_1','residual_block/in1');
    % network=connectLayers(network, 'fuse_a1','residual_block/in2');
    % 
    % 
    % %% flatten layer
    % flat_block=[
    %     convolution2dLayer([1 1],1,"WeightsInitializer",'he',Name="flat_c1",Padding="same");
    % ];
    % network=addLayers(network,flat_block);
    % network=connectLayers(network, 'residual_block','flat_c1');


end