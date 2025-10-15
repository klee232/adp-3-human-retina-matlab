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
classdef patch_histogram_former_layer < nnet.layer.Layer

    properties
        Patch_size
    end
    
    methods

        %% Constructor for the layer
        function layer = patch_histogram_former_layer(name)
           layer.Name = name;
           layer.Description = "patch_histogram_forming_lyr";
        end

        %% Forward pass: patch histogram formation
        function output=predict(layer, input)
            %% retireve dimensional information
            depth_input=size(input,1);
            patch_input=size(input,2);
            patch_num=size(input,4);

            
            %% filter out zero elements
            
            
            
        end
       
    end
end