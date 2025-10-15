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
classdef res_block_chn_multAndadd_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods

        %% Constructor for the layer
        function layer = res_block_chn_multAndadd_layer(numInput,name)
           layer.NumInputs=numInput;
           layer.Name = name;
           layer.Description="channel multiplication in resblock";
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,varargin)
            %% multiply the feature map channel-wisely
            input=varargin;
            chn_propagation=input{1};
            input_feat=input{2};
            group_width=size(chn_propagation,3);
            num_split=size(chn_propagation,4);
            num_cardinal=size(chn_propagation,5);
            output=zeros(size(input_feat,1),size(input_feat,2),group_width,num_split,num_cardinal);
            for i_cardinal=1:num_cardinal
                for i_split=1:num_split
                    current_chn_propagation=chn_propagation(:,:,:,i_split,i_cardinal);
                    start_ind=(i_cardinal-1)*(num_split*group_width)+(i_split-1)*group_width+1;
                    end_ind=(i_cardinal-1)*(num_split*group_width)+(i_split-1)*group_width+group_width;
                    current_input_feat=input_feat(:,:,start_ind:end_ind);
                    output(:,:,:,i_split,i_cardinal)=current_input_feat.*current_chn_propagation;
                end
            end
            output=sum(output,4);
            output=squeeze(output);
            output=dlarray(output,"SSCB");
        end
       
    end
end
