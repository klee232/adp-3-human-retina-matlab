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
classdef patch_transpose_layer < nnet.layer.Layer
    
    methods

        %% Constructor for the layer
        function layer = patch_transpose_layer(name)
           layer.Name = name;
        end

        %% Forward pass: patch formation
        function output=predict(input)
            %% insert input and tranpose each of it
            % extract each patch range
            chn_input=size(input,1);
            row_input=size(input,2);
            col_input=size(input,3);
            num_feature_input=size(input,4);
            output=zeros(chn_input,row_input,col_input,num_feature_input);
            patch_ind=1; 
            for i_feat=1:num_feature_input
                for i_chn=1:chn_input
                    current_patch_input=input(i_chn,:,:,i_feat);
                    current_patch_input=squeeze(current_patch_input);
                    transpose_current_patch_input=transpose(current_patch_input);
                    output(:,:,:,patch_ind)=transpose_current_patch_input;
                    patch_ind=patch_ind+1;
                end
            end
            
        end
       
    end
end
