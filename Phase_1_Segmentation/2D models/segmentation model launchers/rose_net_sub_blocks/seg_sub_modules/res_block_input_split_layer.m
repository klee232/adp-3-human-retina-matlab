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
classdef res_block_input_split_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Num_split
        Num_cardinal
    end
    
    methods

        
        %% Constructor for the layer
        function layer = res_block_input_split_layer(name, num_split, num_cardinal)
           layer.Name = name;
           layer.Description = "split portion of the first half of input";
           layer.Num_split=num_split;
           layer.Num_cardinal=num_cardinal;
        end


        %% Forward pass: patch formation
        function [output]=predict(layer,input)
            %% split the input into two part in the first dimension (SVC and DVC)
            % extract each patch range
            num_feat=size(input,3);
            num_split=layer.Num_split;
            num_cardinal=layer.Num_cardinal;
            group_width=num_feat/(num_split*num_cardinal);
            if floor(group_width)~=group_width
                error('group size not integer');
            end
            output=zeros(size(input,1),size(input,2),group_width,num_split,num_cardinal);
            for i_cardinal=1:num_cardinal
                for i_split=1:num_split
                    start_ind=(i_cardinal-1)*(num_split*group_width)+group_width*(i_split-1)+1;
                    end_ind=(i_cardinal-1)*(num_split*group_width)+group_width*(i_split-1)+group_width;
                    output(:,:,:,i_split,i_cardinal)=input(:,:,start_ind:end_ind,i_split);
                end
            end
            output=dlarray(output,"SSSCB");
        end
       
    end
end
