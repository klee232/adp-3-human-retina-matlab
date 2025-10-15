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

function network =my_SS_Enh()

    % launch the neural network for enhancement
    % setup layers
    batch_input=image3dInputLayer([32 32 32],'Name','input');
    conv1=convolution3dLayer([3 3 3],16,"stride",[1 1 1],"Padding","same");
    relu1=reluLayer;
    batchnorm1=batchNormalizationLayer;
    maxpool1=maxPooling3dLayer([2 2 2],"stride",[2 2 2],"Padding","same");
    conv2=convolution3dLayer([3 3 3],8,"stride", [1 1 1],"Padding","same");
    relu2=reluLayer;
    batchnorm2=batchNormalizationLayer;
    maxpool2=maxPooling3dLayer([2 2 2],"stride",[2 2 2],"Padding","same");
    conv3=convolution3dLayer([3 3 3],8,"stride", [1 1 1],"Padding","same");
    relu3=reluLayer;
    batchnorm3=batchNormalizationLayer;
    maxpool3=maxPooling3dLayer([2 2 2],"stride",[2 2 2],"Padding","same");
    transConv1=transposedConv3dLayer([4 4 4],8,"stride", [2 2 2],"Cropping",[1 1 1;1 1 1]);
    trans_relu1=reluLayer;
    trans_batchnorm1=batchNormalizationLayer;
    transConv2=transposedConv3dLayer([4 4 4],8,"stride",[2 2 2],"Cropping",[1 1 1;1 1 1]);
    trans_relu2=reluLayer;
    trans_batchnorm2=batchNormalizationLayer;
    transConv3=transposedConv3dLayer([4 4 4],16,"stride",[2 2 2],"Cropping",[1 1 1;1 1 1]);
    trans_relu3=reluLayer;
    trans_batchnorm3=batchNormalizationLayer;
    conv4=convolution3dLayer([3 3 3],1,"stride", [1 1 1],"Padding","same");
    clipRelu=clippedReluLayer(1);
    RegLay=regressionLayer();

    % create neural network class 
    network=[...
        input_lay
        batch_input
        conv1
        relu1
        batchnorm1
        maxpool1
        conv2
        relu2
        batchnorm2
        maxpool2
        conv3
        relu3
        batchnorm3
        maxpool3
        transConv1
        trans_relu1
        trans_batchnorm1
        transConv2
        trans_relu2
        trans_batchnorm2
        transConv3
        trans_relu3
        trans_batchnorm3
        conv4
        clipRelu
        RegLay
    ];

end