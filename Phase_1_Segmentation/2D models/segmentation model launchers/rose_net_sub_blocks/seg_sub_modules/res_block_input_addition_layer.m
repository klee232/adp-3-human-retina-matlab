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
classdef res_block_input_addition_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Num_Radix
        Num_Cardinal
    end
    
    methods

        %% Constructor for the layer
        function layer = res_block_input_addition_layer(name,num_radix, num_cardinal)
           layer.Name = name;
           layer.Description="feature addition spatial split attention module in resblock";
           layer.Num_Radix=num_radix;
           layer.Num_Cardinal=num_cardinal;
        end

        %% Forward pass: channel addition
        function [output]=predict(layer,input)
            %% retrieve number of features and number of groups and other dimensional information
            num_radix=layer.Num_Radix;
            num_cardinal=layer.Num_Cardinal;
            num_feat=size(input,3);
            group_size=num_feat/(num_radix*num_cardinal);
            output=zeros(size(input,1),size(input,2),group_size,num_cardinal);

            if floor(group_size)~=group_size
                error('group size is not an integer');
            end

            for i_cardinal=1:num_cardinal
                current_cardinal_output=0;
                for i_group=1:num_radix
                    start_ind=(i_cardinal-1)*(num_radix*group_size)+(i_group-1)*group_size+1;
                    end_ind=(i_cardinal-1)*(num_radix*group_size)+(i_group-1)*group_size+group_size;
                    current_group_feat=input(:,:,start_ind:end_ind,:);
                    current_cardinal_output=current_cardinal_output+current_group_feat;
                end
                output(:,:,:,i_cardinal)=current_cardinal_output;
            end
            output=dlarray(output,"SSCB");

        end
       
    end
end
