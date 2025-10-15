% Created by Kuan-Min Lee
% Created date: Nov. 11th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is built to launch the classifier neural network for
% calssification

% Input Parameter:
% img_depth: depth (channel) number of the image (first dimensional number
% in our case)
% img_row: row number of the image (second dimensional number in our case)
% img_col: column number of the image (third dimensional number in our case)

% Output Parameter:
% network: classification neural network (serial object)



function network=spe_spa_net_launcher(img_depth, img_row, img_col)

    %% launch spectral branch
    %% level 1
    % setup layers
    layers_lv1=[
        % setup image input layer
        image3dInputLayer([img_depth img_row img_col],Normalization="none",Name="input_lyr");
        % convolutional and relu layer
        convolution3dLayer([3 1 1],32,Padding="same",Name='spe_conv_lyr_lv1');
        reluLayer(Name='spe_relu_lyr_lv1');
        ];
    network=dlnetwork(layers_lv1);   
    % setup concatenation layer
    layers_cat_lv1=[
        concatenationLayer(4,2,Name='spe_cat_lyr_lv1');
        maxPooling3dLayer([2 2 1],Stride=2,Name="spe_max_lyr_lv1");
        ];
   network=addLayers(network,layers_cat_lv1);
   network=connectLayers(network, 'spe_relu_lyr_lv1', 'spe_cat_lyr_lv1/in1');
   network=connectLayers(network, 'input_lyr', 'spe_cat_lyr_lv1/in2');

   %% level 2
   % setup convolutional layer
   layers_lv2=[
       convolution3dLayer([3 1 1],64,Padding="same",Name='spe_conv_lyr_xy_lv2');
       reluLayer(Name='spe_relu_lyr_xy_lv2');
       ];
   layers_z_lv2=[
       convolution3dLayer([1 3 3],32,Padding="same",Name='spe_conv_lyr_z_lv2');
       reluLayer(Name='spe_relu_lyr_z_lv2');
   ];
   network=addLayers(network,layers_lv2);
   network=addLayers(network,layers_z_lv2);
   network=connectLayers(network, 'spe_max_lyr_lv1', 'spe_conv_lyr_xy_lv2');
   network=connectLayers(network, 'spe_max_lyr_lv1', 'spe_conv_lyr_z_lv2');
   % setup concatenation layer
   layers_cat_lv2=[
       concatenationLayer(4,3,Name='spe_cat_lyr_lv2');
       % concatenationLayer(4,2,Name='spe_cat_lyr_lv2');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spe_max_lyr_lv2');
       ];
   network=addLayers(network,layers_cat_lv2);
   network=connectLayers(network, 'spe_relu_lyr_xy_lv2', 'spe_cat_lyr_lv2/in1');
   network=connectLayers(network, 'spe_relu_lyr_z_lv2', 'spe_cat_lyr_lv2/in2');
   network=connectLayers(network, 'spe_max_lyr_lv1', 'spe_cat_lyr_lv2/in3');
    
   %% level 3
   % setup convolutional layer
   layers_lv3=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spe_conv_lyr_red_lv3');
       convolution3dLayer([3 1 1],64,Padding="same",Name='spe_conv_lyr_xy_lv3');
       reluLayer(Name='spe_relu_lyr_xy_lv3');
       
       ];
    layers_z_lv3=[
       convolution3dLayer([1 3 3],32,Padding="same",Name='spe_conv_lyr_z_lv3');
       reluLayer(Name='spe_relu_lyr_z_lv3');
    ];
   network=addLayers(network,layers_lv3);
   network=addLayers(network,layers_z_lv3);
   network=connectLayers(network, 'spe_max_lyr_lv2', 'spe_conv_lyr_red_lv3');
   network=connectLayers(network, 'spe_conv_lyr_red_lv3', 'spe_conv_lyr_z_lv3');
   % setup concatenational layer
   layers_cat_lv3=[
       concatenationLayer(4,3,Name='spe_cat_lyr_lv3');
       % concatenationLayer(4,2,Name='spe_cat_lyr_lv3');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spe_max_lyr_lv3');
       ];
   network=addLayers(network,layers_cat_lv3);
   network=connectLayers(network, 'spe_relu_lyr_xy_lv3', 'spe_cat_lyr_lv3/in1');
   network=connectLayers(network, 'spe_relu_lyr_z_lv3', 'spe_cat_lyr_lv3/in2');
   network=connectLayers(network, 'spe_conv_lyr_red_lv3', 'spe_cat_lyr_lv3/in3');

   %% level 4
   % setup convolutional layer
   layers_lv4=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spe_conv_lyr_red_lv4');
       convolution3dLayer([3 1 1],64,Padding="same",Name='spe_conv_lyr_xy_lv4');
       reluLayer(Name='spe_relu_lyr_xy_lv4');
       
    ];
   layers_z_lv4=[
       convolution3dLayer([1 3 3],32,Padding="same",Name='spe_conv_lyr_z_lv4');
       reluLayer(Name='spe_relu_lyr_z_lv4');
   ];
   network=addLayers(network,layers_lv4);
   network=addLayers(network,layers_z_lv4);
   network=connectLayers(network, 'spe_max_lyr_lv3', 'spe_conv_lyr_red_lv4');
   network=connectLayers(network, 'spe_conv_lyr_red_lv4', 'spe_conv_lyr_z_lv4');
   % setup concatenational layer
   layers_cat_lv4=[
       concatenationLayer(4,3,Name='spe_cat_lyr_lv4');
       % concatenationLayer(4,2,Name='spe_cat_lyr_lv4');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spe_max_lyr_lv4');
       ];
   network=addLayers(network,layers_cat_lv4);
   network=connectLayers(network, 'spe_relu_lyr_xy_lv4', 'spe_cat_lyr_lv4/in1');
   network=connectLayers(network, 'spe_relu_lyr_z_lv4', 'spe_cat_lyr_lv4/in2');
   network=connectLayers(network, 'spe_conv_lyr_red_lv4', 'spe_cat_lyr_lv4/in3');
    
   %% level 5
   % setup convolutional layer
   layers_lv5=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spe_conv_lyr_red_lv5');
       convolution3dLayer([3 1 1],96,Padding="same",Name='spe_conv_lyr_xy_lv5');
       reluLayer(Name='spe_relu_lyr_xy_lv5');
    ];
   layers_z_lv5=[
       convolution3dLayer([1 3 3],32,Padding="same",Name='spe_conv_lyr_z_lv5');
       reluLayer(Name='spe_relu_lyr_z_lv5');
   ];
   network=addLayers(network,layers_lv5);
   network=addLayers(network,layers_z_lv5);
   network=connectLayers(network, 'spe_max_lyr_lv4', 'spe_conv_lyr_red_lv5');
   network=connectLayers(network, 'spe_conv_lyr_red_lv5', 'spe_conv_lyr_z_lv5');
   % set concatenational layer
   layers_cat_lv5=[
       concatenationLayer(4,3,Name='spe_cat_lyr_lv5');
       % concatenationLayer(4,2,Name='spe_cat_lyr_lv5');
       convolution3dLayer([1 1 1],1,Padding="same",Name='spe_conv_red_2_lyr_lv5');
       ];
   network=addLayers(network,layers_cat_lv5);
   network=connectLayers(network, 'spe_relu_lyr_xy_lv5', 'spe_cat_lyr_lv5/in1');
   network=connectLayers(network, 'spe_relu_lyr_z_lv5', 'spe_cat_lyr_lv5/in2');
   network=connectLayers(network, 'spe_conv_lyr_red_lv5', 'spe_cat_lyr_lv5/in3');

   %% classification level
   layers_class_lv=[
       globalAveragePooling3dLayer(Name="spe_global_avg_pool");
       ];
    network=addLayers(network,layers_class_lv);
    network=connectLayers(network, 'spe_conv_red_2_lyr_lv5', 'spe_global_avg_pool');


    %% Launch spatial neural network 
    %% level 1
    % setup layers
    layers_lv1=[
        % setup image input layer
        % convolutional and relu layer
        convolution3dLayer([1 3 3],32,Padding="same",Name='spa_conv_lyr_lv1');
        reluLayer(Name='spa_relu_lyr_lv1');
        ];
    network=addLayers(network,layers_lv1); 
    network=connectLayers(network,'input_lyr','spa_conv_lyr_lv1');
    % setup concatenation layer
    layers_cat_lv1=[
        concatenationLayer(4,2,Name='spa_cat_lyr_lv1');
        maxPooling3dLayer([1 2 2],Stride=2,Name="spa_max_lyr_lv1");
        ];
   network=addLayers(network,layers_cat_lv1);
   network=connectLayers(network, 'spa_relu_lyr_lv1', 'spa_cat_lyr_lv1/in1');
   network=connectLayers(network, 'input_lyr', 'spa_cat_lyr_lv1/in2');

   %% level 2
   % setup convolutional layer
   layers_lv2=[
       convolution3dLayer([1 3 3],64,Padding="same",Name='spa_conv_lyr_xy_lv2');
       reluLayer(Name='spa_relu_lyr_xy_lv2');
       ];
   layers_z_lv2=[
       convolution3dLayer([3 1 1],32,Padding="same",Name='spa_conv_lyr_z_lv2');
       reluLayer(Name='spa_relu_lyr_z_lv2');
   ];
   network=addLayers(network,layers_lv2);
   network=addLayers(network,layers_z_lv2);
   network=connectLayers(network, 'spa_max_lyr_lv1', 'spa_conv_lyr_xy_lv2');
   network=connectLayers(network, 'spa_max_lyr_lv1', 'spa_conv_lyr_z_lv2');
   % setup concatenation layer
   layers_cat_lv2=[
       concatenationLayer(4,3,Name='spa_cat_lyr_lv2');
       % concatenationLayer(4,2,Name='spa_cat_lyr_lv2');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spa_max_lyr_lv2');
       ];
   network=addLayers(network,layers_cat_lv2);
   network=connectLayers(network, 'spa_relu_lyr_xy_lv2', 'spa_cat_lyr_lv2/in1');
   network=connectLayers(network, 'spa_relu_lyr_z_lv2', 'spa_cat_lyr_lv2/in2');
   network=connectLayers(network, 'spa_max_lyr_lv1', 'spa_cat_lyr_lv2/in3');
    
   %% level 3
   % setup convolutional layer
   layers_lv3=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spa_conv_lyr_red_lv3');
       convolution3dLayer([1 3 3],64,Padding="same",Name='spa_conv_lyr_xy_lv3');
       reluLayer(Name='spa_relu_lyr_xy_lv3');
       
       ];
    layers_z_lv3=[
       convolution3dLayer([3 1 1],32,Padding="same",Name='spa_conv_lyr_z_lv3');
       reluLayer(Name='spa_relu_lyr_z_lv3');
    ];
   network=addLayers(network,layers_lv3);
   network=addLayers(network,layers_z_lv3);
   network=connectLayers(network, 'spa_max_lyr_lv2', 'spa_conv_lyr_red_lv3');
   network=connectLayers(network, 'spa_conv_lyr_red_lv3', 'spa_conv_lyr_z_lv3');
   % setup concatenational layer
   layers_cat_lv3=[
       concatenationLayer(4,3,Name='spa_cat_lyr_lv3');
       % concatenationLayer(4,2,Name='spa_cat_lyr_lv3');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spa_max_lyr_lv3');
       ];
   network=addLayers(network,layers_cat_lv3);
   network=connectLayers(network, 'spa_relu_lyr_xy_lv3', 'spa_cat_lyr_lv3/in1');
   network=connectLayers(network, 'spa_relu_lyr_z_lv3', 'spa_cat_lyr_lv3/in2');
   network=connectLayers(network, 'spa_conv_lyr_red_lv3', 'spa_cat_lyr_lv3/in3');

   %% level 4
   % setup convolutional layer
   layers_lv4=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spa_conv_lyr_red_lv4');
       convolution3dLayer([1 3 3],64,Padding="same",Name='spa_conv_lyr_xy_lv4');
       reluLayer(Name='spa_relu_lyr_xy_lv4');
       
    ];
   layers_z_lv4=[
       convolution3dLayer([3 1 1],32,Padding="same",Name='spa_conv_lyr_z_lv4');
       reluLayer(Name='spa_relu_lyr_z_lv4');
   ];
   network=addLayers(network,layers_lv4);
   network=addLayers(network,layers_z_lv4);
   network=connectLayers(network, 'spa_max_lyr_lv3', 'spa_conv_lyr_red_lv4');
   network=connectLayers(network, 'spa_conv_lyr_red_lv4', 'spa_conv_lyr_z_lv4');
   % setup concatenational layer
   layers_cat_lv4=[
       concatenationLayer(4,3,Name='spa_cat_lyr_lv4');
       % concatenationLayer(4,2,Name='spa_cat_lyr_lv4');
       maxPooling3dLayer([2 2 2],Stride=2,Name='spa_max_lyr_lv4');
       ];
   network=addLayers(network,layers_cat_lv4);
   network=connectLayers(network, 'spa_relu_lyr_xy_lv4', 'spa_cat_lyr_lv4/in1');
   network=connectLayers(network, 'spa_relu_lyr_z_lv4', 'spa_cat_lyr_lv4/in2');
   network=connectLayers(network, 'spa_conv_lyr_red_lv4', 'spa_cat_lyr_lv4/in3');
    
   %% level 5
   % setup convolutional layer
   layers_lv5=[
       convolution3dLayer([1 1 1],1,Padding="same",Name='spa_conv_lyr_red_lv5');
       convolution3dLayer([1 3 3],96,Padding="same",Name='spa_conv_lyr_xy_lv5');
       reluLayer(Name='spa_relu_lyr_xy_lv5');
    ];
   layers_z_lv5=[
       convolution3dLayer([3 1 1],32,Padding="same",Name='spa_conv_lyr_z_lv5');
       reluLayer(Name='spa_relu_lyr_z_lv5');
   ];
   network=addLayers(network,layers_lv5);
   network=addLayers(network,layers_z_lv5);
   network=connectLayers(network, 'spa_max_lyr_lv4', 'spa_conv_lyr_red_lv5');
   network=connectLayers(network, 'spa_conv_lyr_red_lv5', 'spa_conv_lyr_z_lv5');
   % set concatenational layer
   layers_cat_lv5=[
       concatenationLayer(4,3,Name='spa_cat_lyr_lv5');
       % concatenationLayer(4,2,Name='spa_cat_lyr_lv5');
       convolution3dLayer([1 1 1],1,Padding="same",Name='spa_conv_red_2_lyr_lv5');
       ];
   network=addLayers(network,layers_cat_lv5);
   network=connectLayers(network, 'spa_relu_lyr_xy_lv5', 'spa_cat_lyr_lv5/in1');
   network=connectLayers(network, 'spa_relu_lyr_z_lv5', 'spa_cat_lyr_lv5/in2');
   network=connectLayers(network, 'spa_conv_lyr_red_lv5', 'spa_cat_lyr_lv5/in3');

   %% classification level
   layers_class_lv=[
       globalAveragePooling3dLayer(Name="spa_global_avg_pool");
       ];
    network=addLayers(network,layers_class_lv);
    network=connectLayers(network, 'spa_conv_red_2_lyr_lv5', 'spa_global_avg_pool');


    %% launch the sub modules of the neural network
    % setup final classification layer
    class_lyr=[
       concatenationLayer(4,2,Name='class_cat_lyr');
       fullyConnectedLayer(3,Name='class_fc_lyr');
       softmaxLayer(Name='class_softmax_lyr');
    ];
    network=addLayers(network,class_lyr);
    network=connectLayers(network, 'spe_global_avg_pool','class_cat_lyr/in1');
    network=connectLayers(network, 'spa_global_avg_pool','class_cat_lyr/in2');
    

end