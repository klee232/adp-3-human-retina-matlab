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
classdef res_block_pad_cat_layer < nnet.layer.Layer & nnet.layer.Formattable
    
    methods
        
        %% Constructor for the layer
        function layer = res_block_pad_cat_layer(name,num_input)
           layer.Name = name;
           layer.Description = "auto pad and concatenation of the decoder input";
           layer.NumInputs=num_input;
        end


        %% Forward pass: patch formation
        function [output]=predict(layer,varargin)
            %% retrieve input for the layer
            num_input=layer.NumInputs;
            % grab out files for retrieving largest dimensional information
            % and the sum of number of features
            input=varargin;
            max_row_size=0;
            max_col_size=0;
            num_feat=0;
            for i_input=1:num_input
                current_input=input{i_input};
                current_row_size=size(current_input,1);
                current_col_size=size(current_input,2);
                current_num_feat=size(current_input,3);
                if current_row_size>max_row_size
                    max_row_size=current_row_size;
                end
                if current_col_size>max_col_size
                    max_col_size=current_col_size;
                end
                num_feat=num_feat+current_num_feat;
            end

            output=zeros(max_row_size,max_col_size,num_feat);

            %% pad input if necessary
            accumulated_num_feat=0;
            for i_input=1:num_input
                current_input=input{i_input};
                current_row_size=size(current_input,1);
                current_num_feat=size(current_input,3);
                start_ind=accumulated_num_feat+1;
                end_ind=accumulated_num_feat+current_num_feat;
                pad_size=max_row_size-current_row_size;
                if pad_size>0
                    pad_array=zeros(current_row_size,1,current_num_feat);
                    current_pad_input=cat(2,current_input,pad_array);
                    pad_col_size=size(current_pad_input,2);
                    pad_array=zeros(1,pad_col_size,current_num_feat);
                    current_pad_input=cat(1,current_pad_input,pad_array);
                else
                    current_pad_input=double(current_input);
                end
                output(:,:,start_ind:end_ind)=current_pad_input;
                accumulated_num_feat=accumulated_num_feat+current_num_feat;
            end
            output=dlarray(output,"SSCB");
        end
       
    end
end
