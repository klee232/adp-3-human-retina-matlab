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
classdef res_block_splat_conatenate_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Num_Radix
        Num_Cardinal
    end
    
    methods

        %% Constructor for the layer
        function layer = res_block_splat_conatenate_layer(name)
           layer.Name = name;
           layer.Description="feature concatenation spatial split attention module in resblock";
        end

        %% Forward pass: channel addition
        function [output]=predict(layer,input)
            %% retrieve number of features and number of groups and other dimensional information
            num_cardinal=size(input,4);
            group_width=size(input,3);
            output=zeros(size(input,1),size(input,2),group_width*num_cardinal);
            for i_cardinal=1:num_cardinal
                current_input=input(:,:,:,i_cardinal);
                start_ind=(i_cardinal-1)*group_width+1;
                end_ind=(i_cardinal-1)*group_width+group_width;
                output(:,:,start_ind:end_ind)=current_input;
            end
            output=dlarray(output,"SSCB");
        end
       
    end
end
