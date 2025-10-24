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
classdef fine_block_input_concatenate_layer < nnet.layer.Layer & nnet.layer.Formattable

       
    methods

        %% Constructor for the layer
        function layer = fine_block_input_concatenate_layer(name)
           layer.Name = name;
           layer.Description="fine block input concatenation layer";
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,varargin)
            %% set up propagation coefficients
            input=varargin;
            orig_input=input{1};
            pix_input=input{2};
            cen_input=input{3};
            output=cat(3,orig_input,pix_input,cen_input);
            output=dlarray(output,"SSSCB");

        end
       
    end
end
