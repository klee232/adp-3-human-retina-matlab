% Created by Kuan-Min Lee
% Created date: Dec. 12th, 2023 
% All rights reserved to Leelab.ai (abandoned)

% Brief User Introduction:
% This is the launcher for the neural network for Sabina's
% EnhVess Network

% Input Parameter: None
% Output Parameter:
% EnhVess Neural Network

function Enhanced_img=SS_Enh(input_image)

    % launch the neural network for enhancement
    network=[
        imageInputLayer([32 32 32])
        convolution3dLayer([3 3 3],16,"stride",[1 1 1],Padding="same")
        reluLayer
        batchNormalizationLayer
        maxPooling3dLayer([2 2 2],"Stride",[2 2 2],"Padding","same")
        convolution3dLayer([3 3 3],8,"stride",[1 1 1],Padding="same")
        reluLayer
        batchNormalizationLayer
        maxPooling3dLayer([2 2 2],"Stride",[2 2 2],"Padding","same")
        convolution3dLayer([3 3 3],8,"stride",[1 1 1],Padding="same")
        reluLayer
        batchNormalizationLayer
        maxPooling3dLayer([2 2 2],"Stride",[2 2 2],"Padding","same")
        transposedConv3dLayer([4 4 4],8,"Stride",[2 2 2],"Cropping",[1 1 1;1 1 1])
        reluLayer
        batchNormalizationLayer
        transposedConv3dLayer([4 4 4],8,"Stride",[2 2 2],"Cropping",[1 1 1;1 1 1])
        reluLayer
        batchNormalizationLayer
        transposedConv3dLayer([4 4 4],16,"Stride",[2 2 2],"Cropping",[1 1 1;1 1 1])
        reluLayer
        batchNormalizationLayer
        convolution3dLayer([3 3 3],"stride",[1 1 1],"Padding","same")
        clippedReluLayer(1)
        regressionout
    ];

    % Output the Enhanced Image
    Enhanced_img = activations(network, log_input_image, 'regressionoutput', 'executionenvironment', 'cpu');


end