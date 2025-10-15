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
classdef patch_embedded_layer < nnet.layer.Layer

    properties
        Patch_size
    end
    
    methods

        %% Constructor for the layer
        function layer = patch_former_layer(name,input)
           layer.Name = name;
           layer.Description = "patch_embedded_lyr";
           layer.Patch_size = floor(size(input,2)/3);
        end

        %% Forward pass: patch formation
        function output=predict(layer, input)
            %% insert input and form patches
            % retireve dimensional information
            depth_input=size(input,1);
            row_input=size(input,2);
            col_input=size(input,3);

            % if the the dimension is not divideable by patch size, pad the
            % input image
            pad_row_size=mod(row_input,layer.Patch_size);
            pad_col_size=mod(col_input,layer.Patch_size);
            if pad_row_size~=0 || pad_col_size~=0
                pad_input=zeros(depth_input,(row_input+pad_row_size),(col_input+pad_col_size));
                for i_dep=1:depth_input
                    current_input=squeeze(input(i_dep,:,:));
                    pad_current_input=padarray(current_input,[pad_row_size pad_col_size],0,'post');
                    pad_input(i_dep,:,:)=pad_current_input;
                end
            else
                pad_input=input;
            end

            % create output storage
            pad_input_row=size(pad_input,2);
            pad_input_col=size(pad_input,3);
            pad_input_num_feat=size(pad_input,4);
            factor_row=pad_input_row/layer.Patch_size;
            factor_col=pad_input_col/layer.Patch_size;
            output=zeros(depth_input,layer.Patch_size, layer.Patch_size, (factor_row*factor_col)*pad_input_num_feat);

            % extract each patch range
            patch_ind=1;
            for i_chn=1:pad_input_num_feat
                for i_row=1:layer.Patch_size:pad_input_row
                    for i_col=1:layer.Patch_size:pad_input_col
                        current_patch_input=pad_input(:,i_row:(i_row+layer.Patch_size),i_col:(i_col+layer.Patch_size),i_chn);
                        output(:,:,:,patch_ind)=current_patch_input;
                        patch_ind=patch_ind+1;
                    end
                end
            end
            
        end
       
    end
end