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
classdef res_block_3d_chn_multiplication_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods

        %% Constructor for the layer
        function layer = res_block_3d_chn_multiplication_layer(numInput,name)
           layer.NumInputs=numInput;
           layer.Name = name;
           layer.Description="channel multiplication in resblock";
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,varargin)
            %% multiply the feature map channel-wisely
            input=varargin;
            input_feat=input{1};
            input_chn=input{2};
            num_feat=size(input{1},4);
            output=zeros(size(input{1}));
            for i_feat=1:num_feat
                output(:,:,:,i_feat,:)=input_feat(:,:,:,i_feat,:).*input_chn(i_feat,:);
            end
            output=dlarray(output,"SSSCB");
        end
       
    end
end
