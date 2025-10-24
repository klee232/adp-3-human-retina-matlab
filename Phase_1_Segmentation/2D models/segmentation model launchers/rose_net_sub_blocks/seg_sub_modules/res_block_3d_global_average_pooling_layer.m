% Created by Kuan-Min Lee
% Created date: Dec. 5th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This is the customized neural network layer which serves as the function
% of patch mean removal 

% Input Parameter:
% patch_size: size of patch for patch formation
% input: original input feature

% Output Parameter:
% output: processed array (4D array with size of [depth, patch_size,
% patch_size, number_patch])
classdef res_block_3d_global_average_pooling_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods

        %% Constructor for the layer
        function layer = res_block_3d_global_average_pooling_layer(name)
           layer.Name = name;
           layer.Description="global average pooling in resblock";
        end

        %% Forward pass: patch formation
        function [output]=predict(layer,input)
            %% average in the first three dimension (layer, row, column)
            output=mean(input,[1 2 3]);
        end
       
    end
end
