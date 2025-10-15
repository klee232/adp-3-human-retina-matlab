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



function network=mean_removal_pca_net_en_face_launcher(img_row, img_col, num_features, num_classes)

    %% setup layers in the network

    %% level 1
    % setup layers
    % mean_removal_lyr_lv1=mean_removal_layer('mean_removal_lyr_lv1',3);
    % pca_conv_lyr_lv1=pca_convolution_layer('pca_conv_lyr_lv1',input, 1, 3, 3, 2);
    layers_lv1=[
        % setup image input layer
        imageInputLayer([img_row img_col num_features],Normalization="none",Name="input_lyr");
        % convolutional and relu layer
        % mean_removal_lyr_lv1;
        % pca_conv_lyr_lv1;
        convolution2dLayer([3 3],16,Padding="same",Name='conv_lyr_lv1');
        reluLayer(Name='relu_lyr_lv1');
        ];
    network=dlnetwork(layers_lv1);   
    % setup concatenation layer
    layers_cat_lv1=[
        concatenationLayer(3,2,Name='cat_lyr_lv1');
        maxPooling2dLayer(2,Stride=2,Name="max_lyr_lv1");
        ];
   network=addLayers(network,layers_cat_lv1);
   network=connectLayers(network, 'relu_lyr_lv1', 'cat_lyr_lv1/in1');
   network=connectLayers(network, 'input_lyr', 'cat_lyr_lv1/in2');

   %% level 2
   % setup convolutional layer
   % mean_removal_lyr_xy_lv2=mean_removal_layer('mean_removal_lyr_xy_lv2',3);
   % pca_conv_lyr_xy_lv2=pca_convolution_layer('pca_conv_lyr_xy_lv2',input, 1, 3, 3, 4);
   layers_lv2=[
       % mean_removal_lyr_xy_lv2;
       % pca_conv_lyr_xy_lv2;
       convolution2dLayer([3 3],64,Padding="same",Name='conv_lyr_xy_lv2');
       reluLayer(Name='relu_lyr_xy_lv2');
       ];
   % mean_removal_lyr_z_lv2=mean_removal_layer('mean_removal_lyr_z_lv2',3);
   % pca_conv_lyr_z_lv2=pca_convolution_layer('pca_conv_lyr_z_lv2',input, 3, 1, 1, 2);
   layers_z_lv2=[
       % mean_removal_lyr_z_lv2;
       % pca_conv_lyr_z_lv2;
       convolution2dLayer([1 1],32,Padding="same",Name='conv_lyr_z_lv2');
       reluLayer(Name='relu_lyr_z_lv2');
   ];
   network=addLayers(network,layers_lv2);
   network=addLayers(network,layers_z_lv2);
   network=connectLayers(network, 'max_lyr_lv1', 'conv_lyr_xy_lv2');
   network=connectLayers(network, 'max_lyr_lv1', 'conv_lyr_z_lv2');
   % setup concatenation layer
   layers_cat_lv2=[
       concatenationLayer(3,3,Name='cat_lyr_lv2');
       maxPooling2dLayer(2,Stride=2,Name='max_lyr_lv2');
       ];
   network=addLayers(network,layers_cat_lv2);
   network=connectLayers(network, 'relu_lyr_xy_lv2', 'cat_lyr_lv2/in1');
   network=connectLayers(network, 'relu_lyr_z_lv2', 'cat_lyr_lv2/in2');
   network=connectLayers(network, 'conv_lyr_xy_lv2', 'cat_lyr_lv2/in3');
    
   %% level 3
   % setup convolutional layer
   % mean_removal_lyr_red_lv3=mean_removal_layer('mean_removal_lyr_red_lv3',3);
   % pca_conv_lyr_red_lv3=pca_convolution_layer('pca_conv_lyr_red_lv3',input, 1, 1, 1, 1);
   % mean_removal_lyr_xy_lv3=mean_removal_layer('mean_removal_lyr_xy_lv3',3);
   % pca_conv_lyr_xy_lv3=pca_convolution_layer('pca_conv_lyr_xy_lv3',input, 1, 3, 3, 4);
   layers_lv3=[
       % mean_removal_lyr_red_lv3;
       % pca_conv_lyr_red_lv3;
       convolution2dLayer([1 1],16,Padding="same",Name='conv_lyr_red_lv3');
       % mean_removal_lyr_xy_lv3;
       % pca_conv_lyr_xy_lv3
       convolution2dLayer([3 3],64,Padding="same",Name='conv_lyr_xy_lv3');
       reluLayer(Name='relu_lyr_xy_lv3');
       ];
    % mean_removal_lyr_z_lv3=mean_removal_layer('mean_removal_lyr_z_lv3',3);
    % pca_conv_lyr_z_lv3=pca_convolution_layer('pca_conv_lyr_z_lv3',input, 3, 1, 1, 2);
    layers_z_lv3=[
       % mean_removal_lyr_z_lv3;
       % pca_conv_lyr_z_lv3
       convolution2dLayer([1 1],32,Padding="same",Name='conv_lyr_z_lv3');
       reluLayer(Name='relu_lyr_z_lv3');
    ];
   network=addLayers(network,layers_lv3);
   network=addLayers(network,layers_z_lv3);
   network=connectLayers(network, 'max_lyr_lv2', 'conv_lyr_red_lv3');
   network=connectLayers(network, 'conv_lyr_red_lv3', 'conv_lyr_z_lv3');
   % setup concatenational layer
   layers_cat_lv3=[
       concatenationLayer(3,3,Name='cat_lyr_lv3');
       maxPooling2dLayer([2 2],Stride=2,Name='max_lyr_lv3');
       ];
   network=addLayers(network,layers_cat_lv3);
   network=connectLayers(network, 'relu_lyr_xy_lv3', 'cat_lyr_lv3/in1');
   network=connectLayers(network, 'relu_lyr_z_lv3', 'cat_lyr_lv3/in2');
   network=connectLayers(network, 'conv_lyr_red_lv3', 'cat_lyr_lv3/in3');

   %% level 4
   % setup convolutional layer
   % mean_removal_lyr_red_lv4=mean_removal_layer('mean_removal_lyr_red_lv4',3);
   % pca_conv_lyr_red_lv4=pca_convolution_layer('pca_conv_lyr_red_lv4',input, 1, 1, 1, 1);
   % mean_removal_lyr_xy_lv4=mean_removal_layer('mean_removal_lyr_xy_lv4',3);
   % pca_conv_lyr_xy_lv4=pca_convolution_layer('pca_conv_lyr_xy_lv4',input, 1, 3, 3, 4);
   layers_lv4=[
       % mean_removal_lyr_red_lv4;
       % pca_conv_lyr_red_lv4;
       convolution2dLayer([1 1],16,Padding="same",Name='conv_lyr_red_lv4');
       % mean_removal_lyr_xy_lv4;
       % pca_conv_lyr_xy_lv4;
       convolution2dLayer([3 3],64,Padding="same",Name='conv_lyr_xy_lv4');
       reluLayer(Name='relu_lyr_xy_lv4');
    ];
   % mean_removal_lyr_z_lv4=mean_removal_layer('mean_removal_lyr_z_lv4',3);
   % pca_conv_lyr_z_lv4=pca_convolution_layer('pca_conv_lyr_z_lv4',input, 3, 1, 1, 2);
   layers_z_lv4=[
       % mean_removal_lyr_z_lv4;
       % pca_conv_lyr_z_lv4;
       convolution2dLayer([1 1],32,Padding="same",Name='conv_lyr_z_lv4');
       reluLayer(Name='relu_lyr_z_lv4');
   ];
   network=addLayers(network,layers_lv4);
   network=addLayers(network,layers_z_lv4);
   network=connectLayers(network, 'max_lyr_lv3', 'conv_lyr_red_lv4');
   network=connectLayers(network, 'conv_lyr_red_lv4', 'conv_lyr_z_lv4');
   % setup concatenational layer
   layers_cat_lv4=[
       concatenationLayer(3,3,Name='cat_lyr_lv4');
       maxPooling2dLayer([2 2],Stride=2,Name='max_lyr_lv4');
       ];
   network=addLayers(network,layers_cat_lv4);
   network=connectLayers(network, 'relu_lyr_xy_lv4', 'cat_lyr_lv4/in1');
   network=connectLayers(network, 'relu_lyr_z_lv4', 'cat_lyr_lv4/in2');
   network=connectLayers(network, 'conv_lyr_red_lv4', 'cat_lyr_lv4/in3');
    
   %% level 5
   % setup convolutional layer
   % mean_removal_lyr_red_lv5=mean_removal_layer('mean_removal_lyr_red_lv5',3);
   % pca_conv_lyr_red_lv5=pca_convolution_layer('pca_conv_lyr_red_lv5',input, 1, 1, 1, 1);
   % mean_removal_lyr_xy_lv5=mean_removal_layer('mean_removal_lyr_xy_lv5',3);
   % pca_conv_lyr_xy_lv5=pca_convolution_layer('pca_conv_lyr_xy_lv5',input, 1, 3, 3, 6);
   layers_lv5=[
       % mean_removal_lyr_red_lv5;
       % pca_conv_lyr_red_lv5
       convolution2dLayer([1 1],16,Padding="same",Name='conv_lyr_red_lv5');
       % mean_removal_lyr_xy_lv5;
       % pca_conv_lyr_xy_lv5;
       convolution2dLayer([3 3],256,Padding="same",Name='conv_lyr_xy_lv5');
       reluLayer(Name='relu_lyr_xy_lv5');
    ];
   % mean_removal_lyr_z_lv5=mean_removal_layer('mean_removal_lyr_z_lv5',3);
   % pca_conv_lyr_z_lv5=pca_convolution_layer('pca_conv_lyr_z_lv5',input, 3, 1, 1, 3);
   layers_z_lv5=[
       % mean_removal_lyr_z_lv5;
       % pca_conv_lyr_z_lv5;
       convolution2dLayer([1 1],48,Padding="same",Name='conv_lyr_z_lv5');
       reluLayer(Name='relu_lyr_z_lv5');
   ];
   network=addLayers(network,layers_lv5);
   network=addLayers(network,layers_z_lv5);
   network=connectLayers(network, 'max_lyr_lv4', 'conv_lyr_red_lv5');
   network=connectLayers(network, 'conv_lyr_red_lv5', 'conv_lyr_z_lv5');
   % set concatenational layer
   % mean_removal_lyr_red_2_lv5=mean_removal_layer('mean_removal_lyr_red_2_lv5',3);
   % pca_conv_lyr_red_2_lv5=pca_convolution_layer('conv_lyr_red_2_lv5',input, 1, 1, 1, 1);
   layers_cat_lv5=[
       concatenationLayer(3,3,Name='cat_lyr_lv5');
       % mean_removal_lyr_red_2_lv5;
       % pca_conv_lyr_red_2_lv5;
       convolution2dLayer([1 1],16,Padding="same",Name='conv_red_2_lyr_lv5');
       ];
   network=addLayers(network,layers_cat_lv5);
   network=connectLayers(network, 'relu_lyr_xy_lv5', 'cat_lyr_lv5/in1');
   network=connectLayers(network, 'relu_lyr_z_lv5', 'cat_lyr_lv5/in2');
   network=connectLayers(network, 'conv_lyr_red_lv5', 'cat_lyr_lv5/in3');

   %% classification level
   layers_class_lv=[
       maxPooling2dLayer([2 2],Stride=2,Name="max_lyr_lv5");
       fullyConnectedLayer(num_classes);
       softmaxLayer(Name='class_softmax_lyr');
       ];
    network=addLayers(network,layers_class_lv);
    network=connectLayers(network, 'conv_red_2_lyr_lv5', 'max_lyr_lv5');

end