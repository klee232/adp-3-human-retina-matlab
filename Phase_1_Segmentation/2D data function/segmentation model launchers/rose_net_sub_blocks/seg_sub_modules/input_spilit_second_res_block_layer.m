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
classdef input_spilit_second_res_block_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods


        %% Constructor for the layer
        function layer = input_spilit_second_res_block_layer(name)
           layer.Name = name;
           layer.Description = "split portion of the second half of input";
        end

        
        %% Forward pass: patch formation
        function [output]=predict(layer,input)
            %% split the input into two part in the first dimension (SVC and DVC)
            % extract each patch range
            num_feat=size(input,3);
            output=input(:,:,(floor(num_feat/2)+1):end,:);
        end
       
    end
end
