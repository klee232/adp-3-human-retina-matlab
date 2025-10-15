% Created by Kuan-Min Lee
% Created date: Mar. 9th, 2025
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
classdef fine_block_propagation_layer < nnet.layer.Layer & nnet.layer.Formattable

    %% parameters (not learnable)
    properties (Learnable)
        Weights;
    end

    properties
        Kernel_size;
    end

    methods

        %% Constructor for the layer
        function layer = fine_block_propagation_layer(num_input,kernel_size, name)
           %% setup unleanrable parameters
           layer.Name = name;
           layer.Description="propagation layer";
           layer.NumInputs=num_input;
           layer.Kernel_size=kernel_size;
           layer.Weights=randn(1,1,kernel_size*kernel_size);

        end


        %% Forward pass: adaptive aggregation
        function [output]=predict(layer,varargin)
            %% grab out input
            input=varargin;
            coarse_input=input{1};
            feat_map=input{2};


            %% grab out necessary parameters
            kernel_size=layer.Kernel_size;


            %% conduct adaptive aggregration
            pad_coarse_input=padarray(extractdata(coarse_input),[floor(kernel_size/2) floor(kernel_size/2)],0,'both');
            patch_pad_coarse_input=im2col(pad_coarse_input,[kernel_size kernel_size],'sliding');
            row_size=size(feat_map,1);
            col_size=size(feat_map,2);
            reshaped_feat_map=reshape(extractdata(feat_map),row_size*col_size,kernel_size*kernel_size)';
            output=sum(patch_pad_coarse_input.*reshaped_feat_map,1);
            output=reshape(output,row_size,col_size);
            output=dlarray(output,"SSCB");

        end
       
    end
end
