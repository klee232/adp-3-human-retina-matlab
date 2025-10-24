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
classdef coarse_block_output_conatenate_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods

        %% Constructor for the layer
        function layer = coarse_block_output_conatenate_layer(name,num_input)
           layer.Name = name;
           layer.Description="coarse neurel network output concatenation";
           layer.NumInputs=num_input;
        end

        %% Forward pass: channel addition
        function [output]=predict(layer,varargin)
            %% retrieve number of features and number of groups and other dimensional information
            input=varargin;
            pixel_input=input{1};
            cent_input=input{2};
            output=cat(3,pixel_input,cent_input);
            output=dlarray(output,"SSCB");
        end
       
    end
end
