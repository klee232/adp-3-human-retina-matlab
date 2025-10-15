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
classdef res_block_split_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Num_split
    end
    
    methods

        
        %% Constructor for the layer
        function layer = res_block_split_layer(name, num_split)
           layer.Name = name;
           layer.Description = "split portion of the first half of input";
           layer.Num_split=num_split;
        end


        %% Forward pass: patch formation
        function [output]=predict(layer,input)
            %% split the input into two part in the first dimension (SVC and DVC)
            % extract each patch range
            num_feat=size(input,3);
            num_split=layer.Num_split;
            group_width=num_feat/num_split;
            if ~isinteger(group_width)
                error('group size not integer');
            end
            output=zeros(size(input,1),size(input,2),group_width,num_split);
            for i_split=1:num_split
                start_ind=group_width*(i_split-1)+i_split;
                end_ind=group_width*(i_split-1)+group_width;
                output(:,:,:,i_split)=input(:,:,start_ind:end_ind,:);
            end
        end
       
    end
end
