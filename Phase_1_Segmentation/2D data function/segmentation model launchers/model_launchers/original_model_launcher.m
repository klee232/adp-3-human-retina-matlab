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


function [network]=original_model_launcher(img_depth, img_row, img_col)

    %% block 1
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu 
    layerb1m1=[
        image3dInputLayer([img_depth img_row img_col 1],Normalization="none",Name="input_lyr");
        convolution3dLayer([1 3 3],64,Padding="same",Name='b1m1c');
        batchNormalizationLayer(Name='b1m1b');
        reluLayer(Name='b1m1a');
        ];
    network=dlnetwork(layerb1m1);   

    % module layer 2: 3*3 convolutional layer + batchnromalization + relu
    layerb1m2=[
        convolution3dLayer([1 3 3],64,Padding="same",Name='b1m2c');
        batchNormalizationLayer(Name='b1m2b');
        reluLayer(Name='b1m2a');
    ];
    network=addLayers(network,layerb1m2);
    network=connectLayers(network, 'b1m1a', 'b1m2c');

    % additional layer
    layerb1add=[
       additionLayer(2,Name='b1m1add');
       ];
    network=addLayers(network,layerb1add);
    network=connectLayers(network, 'b1m1a', 'b1m1add/in1');
    network=connectLayers(network, 'b1m2a', 'b1m1add/in2');

    % max pooling layer (bulk)
    layerb1maxb=[
       maxPooling3dLayer([1 2 2],Stride=[1 2 2],Name='b1m1maxb');
       ];
    network=addLayers(network,layerb1maxb);
    network=connectLayers(network, 'b1m1add', 'b1m1maxb');

    % max pooling layer (detail)
    layerb1maxd=[
       maxPooling3dLayer([1 2 2],Stride=[1 2 2], Name='b1m1maxd');
    ];
    network=addLayers(network,layerb1maxd);
    network=connectLayers(network, 'b1m2a', 'b1m1maxd');
    

    %% bottleneck layer 
    % module block 1
    layerbt1m1=[
        convolution3dLayer([1 3 3],128,Padding="same",Name='bt1m1c1');
        batchNormalizationLayer(Name='bt1m1b1');
        reluLayer(Name='bt1m1a1');
        ];
    network=addLayers(network,layerbt1m1);
    network=connectLayers(network, 'b1m1maxb', 'bt1m1c1');

    % module block 2
    layerbt1m2=[
        convolution3dLayer([1 3 3],128,Padding="same",Name='bt1m2c1');
        batchNormalizationLayer(Name='bt1m2b1');
        reluLayer(Name='bt1m2a1');
        ];
    network=addLayers(network,layerbt1m2);
    network=connectLayers(network, 'bt1m1a1', 'bt1m2c1');

    % module block 3
    layerbt1m3=[
        convolution3dLayer([1 3 3],128,Padding="same",Name='bt1m3c1');
        batchNormalizationLayer(Name='bt1m3b1');
        reluLayer(Name='bt1m3a1');
        ];
    network=addLayers(network,layerbt1m3);
    network=connectLayers(network, 'bt1m2a1', 'bt1m3c1');


     %% detail branch block
    % module block 1
    layerdb1m1=[
        convolution3dLayer([1 3 3],16,Padding="same",Name='db1m1c1');
        convolution3dLayer([1 3 3],16,Padding="same",Name='db1m1c2');
    ];
    network=addLayers(network,layerdb1m1);
    network=connectLayers(network, 'b1m1maxd', 'db1m1c1');

    % module block 2
    layerdb1m2b1=[
        convolution3dLayer([1 3 3],16,Padding="same",DilationFactor=[1 2 2],Name='db1m2dc1');
    ];
    layerdb1m2b2=[
        convolution3dLayer([1 5 5],16,Padding="same",DilationFactor=[1 1 1],Name='db1m2dc2');
    ];
    layerdb1m2cat=[
        concatenationLayer(4,2,Name='db1m2cat');
    ];
    network=addLayers(network,layerdb1m2b1);
    network=addLayers(network,layerdb1m2b2);
    network=addLayers(network,layerdb1m2cat);
    network=connectLayers(network, 'db1m1c2', 'db1m2dc1');
    network=connectLayers(network, 'db1m1c2', 'db1m2dc2');
    network=connectLayers(network, 'db1m2dc1', 'db1m2cat/in1');
    network=connectLayers(network, 'db1m2dc2', 'db1m2cat/in2');

    % module block 3
    layerdb1m3b1=[
        convolution3dLayer([1 3 3],16,Padding="same",DilationFactor=[1 4 4],Name='db1m3dc1');
    ];
    layerdb1m3b2=[
        convolution3dLayer([1 5 5],16,Padding="same",DilationFactor=[1 3 3],Name='db1m3dc2');
    ];
    layerdb1m3b3=[
        convolution3dLayer([1 7 7],16,Padding="same",DilationFactor=[1 2 2],Name='db1m3dc3');
    ];
    layerdb1m3b4=[
        convolution3dLayer([1 9 9],16,Padding="same",DilationFactor=[1 1 1],Name='db1m3dc4');
    ];
    layerdb1m3cat=[
        concatenationLayer(4,4,Name='db1m3cat');
    ];
    network=addLayers(network,layerdb1m3b1);
    network=addLayers(network,layerdb1m3b2);
    network=addLayers(network,layerdb1m3b3);
    network=addLayers(network,layerdb1m3b4);
    network=addLayers(network,layerdb1m3cat);
    network=connectLayers(network, 'db1m2cat', 'db1m3dc1');
    network=connectLayers(network, 'db1m2cat', 'db1m3dc2');
    network=connectLayers(network, 'db1m2cat', 'db1m3dc3');
    network=connectLayers(network, 'db1m2cat', 'db1m3dc4');
    network=connectLayers(network, 'db1m3dc1', 'db1m3cat/in1');
    network=connectLayers(network, 'db1m3dc2', 'db1m3cat/in2');
    network=connectLayers(network, 'db1m3dc3', 'db1m3cat/in3');
    network=connectLayers(network, 'db1m3dc4', 'db1m3cat/in4');

    % module block 4
    layerdb1m4=[
        transposedConv3dLayer([1 2 2],64,"Stride",[1 2 2],Name='db1m4t'); 
        ];
    network=addLayers(network,layerdb1m4);
    network=connectLayers(network, 'db1m3cat', 'db1m4t');


    %% transposed layer 1
    % module block 1
    layertb1m1=[
        transposedConv3dLayer([1 2 2],64,"Stride",[1 2 2],Name='tb1m1t'); 
        ];
    network=addLayers(network,layertb1m1);
    network=connectLayers(network, 'bt1m3a1', 'tb1m1t');

    % module block 2
    layertb1m2=[
        concatenationLayer(4,2,Name='tb1m2cat');
        convolution3dLayer([1 3 3],64,Padding="same",Name='tb1m2c1');
        batchNormalizationLayer(Name='tb1m2b1');
        reluLayer(Name='tb1m2a1');
        ];
    network=addLayers(network,layertb1m2);
    network=connectLayers(network, 'tb1m1t', 'tb1m2cat/in1');
    network=connectLayers(network, 'db1m4t', 'tb1m2cat/in2');

    % module block 3
    layertb1m3=[
        convolution3dLayer([1 3 3],64,Padding="same",Name='tb1m3c1');
        batchNormalizationLayer(Name='tb1m3b1');
        reluLayer(Name='tb1m3a1');
        ];
    network=addLayers(network,layertb1m3);
    network=connectLayers(network, 'tb1m2a1', 'tb1m3c1');

   % additional layer
    layertb1add=[
       additionLayer(2,Name='tb1add');
       ];
    network=addLayers(network,layertb1add);
    network=connectLayers(network, 'tb1m3a1', 'tb1add/in1');
    network=connectLayers(network, 'tb1m2a1', 'tb1add/in2');


    %% flatten block 1
    % module layer 1: 3*3 convolutional layer + batchnromalization + relu 
    layerfb1m1=[
        convolution3dLayer([1 1 1],1,Padding="same",Name='fb1m1c1');
        batchNormalizationLayer(Name='fb1m1b1');
        sigmoidLayer(Name='fb1m1a1');
        ];
    network=addLayers(network,layerfb1m1);
    network=connectLayers(network, 'tb1add', 'fb1m1c1');

    
end