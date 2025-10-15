% Created by Kuan-Min Lee
% (inspired by Sabina Stefan's Work of Deep
% learning ttoolbox for automated enhancement, segmenttation, and graphing
% of cortical opptical coherence tomography microangiograms)
% Created date: Dec. 12th, 2023
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This is the second version of the launcher of EnhVess Network
% EnhVess Network

% Input Parameter: None
% Output Parameter:
% EnhVess Neural Network

function lgraph =my_SS_Seg()
    % launch the neural network for enhancement
    % construct hidden layer part
    batch_input=image3dInputLayer([12 16 16],'Name','input');
    conv1=convolution3dLayer([3 3 3],64,"stride",[1 1 1],"Padding","same",'Name','conv1');
    relu1=reluLayer('Name','relu1');
    batchnorm1=batchNormalizationLayer('Name','batchnorm1');
    maxpool1=maxPooling3dLayer([2 2 2],"stride",[2 2 2],"Padding","same",'Name','maxpool1');
    conv2=convolution3dLayer([3 3 3],64,"stride", [1 1 1],"Padding","same",'Name','conv2');
    relu2=reluLayer('Name','relu2');
    batchnorm2=batchNormalizationLayer('Name','batchnorm2');
    transConv1=transposedConv3dLayer([4 4 4],64,"stride", [2 2 2],"Cropping",[1 1 1;1 1 1],'Name','transConv1');
    layers=[...
        batch_input
        conv1
        relu1
        batchnorm1
        maxpool1
        conv2
        relu2
        batchnorm2
        transConv1
        ];
    lgraph=layerGraph(layers);
    % Concatenate hidden layers and skipped connection
    concate_1=concatenationLayer(4,2,'Name','concate_1');
    lgraph=addLayers(lgraph,concate_1);
    lgraph=connectLayers(lgraph,'batchnorm1','concate_1/in1');
    lgraph=connectLayers(lgraph,'transConv1','concate_1/in2');

    % construct later part
    conv3=convolution3dLayer([3 3 3],2,"stride", [1 1 1],"Padding","same",'Name','conv3');
    relu3=reluLayer('Name','relu3');
    batchnorm3=batchNormalizationLayer('Name','batchnorm3');
    softmax=softmaxLayer('Name','softmax');
    classes=["background","vessel"];
    classes=categorical(classes);
    classify=tverskyPixelClassificationLayer('classify', 0.3, 0.7);
%     classify=dicePixelClassificationLayer('Classes',classes,'Name','classify');
    lgraph=addLayers(lgraph,conv3);
    lgraph=addLayers(lgraph,relu3);
    lgraph=addLayers(lgraph,batchnorm3);
    lgraph=addLayers(lgraph,softmax);
    lgraph=addLayers(lgraph,classify);
    lgraph=connectLayers(lgraph,'concate_1','conv3');
    lgraph=connectLayers(lgraph,'conv3','relu3');
    lgraph=connectLayers(lgraph,'relu3','batchnorm3');
    lgraph=connectLayers(lgraph,'batchnorm3','softmax');
    lgraph=connectLayers(lgraph,'softmax','classify');

    % create forward function
        Enhanced_img = activations(network, log_input_image, 'regressionoutput', 'executionenvironment', 'cpu');

    feat_conv1=activations(lgraph,)



end