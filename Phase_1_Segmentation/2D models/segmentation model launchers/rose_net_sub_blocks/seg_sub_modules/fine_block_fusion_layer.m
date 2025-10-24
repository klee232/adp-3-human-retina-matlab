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
classdef fine_block_fusion_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods

        %% Constructor for the layer
        function layer = fine_block_fusion_layer(numInput,name)
           layer.NumInputs=numInput;
           layer.Name = name;
           layer.Description="fine stage fusion";
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,varargin)
            %% multiply the feature map channel-wisely
            % num_input=layer.NumInputs;
            input=varargin;
            fine_p=input{1};
            fine_c=input{2};
            output=max(fine_p,fine_c);
            output=dlarray(output,"SSCB");
        end
       
    end
end
