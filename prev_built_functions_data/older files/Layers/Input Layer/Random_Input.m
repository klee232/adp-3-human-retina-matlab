% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter
% patch_num: the number of patch (integer value)
% patch_size: the size of the patch (integer value or a vector)
% Input Parameter:
% input_data: the data that is intended to be patched (multi-dimension numerical array)
% Output Parameter
% patched_data: the patched data (multi-dimension numerical array)

classdef Random_Input < nnet.layer.Layer 
    % define the paramters for convolutional Random_Patched_Input layer 
    properties
        % parameters for convolutional Patched Input layer
        Patch_Num
        Patch_Size
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Random_Input (name,num_patch,patch_size)
            % layer = Random_Input creates a random patched input layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Random Patched Input"; % description of current layer
            % Parameter parts
            layer.Patch_Num=num_patch; % number of patch
            layer.Patch_Size=patch_size; % Patch size
        end
        % forward function of the layer
        function patched_data = predict(layer,input_data)
            input_data=gpuArray(input_data);
            % Input Parameter: 
            % layer
            % input_data: the data that is intended to be patched (multi-dimension
            % numerical array)
            % patch_num: the number of patch (integer value)
            % patch_size: the size of the patch (integer value or a vector)
            % Output Parameter:
            % patched_data: the patched data (multi-dimension numerical array)
            patch_num=layer.Patch_Num;
            patch_size=layer.Patch_Size;

            % Check the Input Argument
            % check the dimension of the input data
            % if the dimension of the input data is less than 2, send out an error
            % message
            num_dims=ndims(input_data);
            if num_dims<2
                error("The input needs to be at least a 2D input. Please double check the dimension of the input")
            end
            % receive the size of the input data
            [row_data,col_data,chn_data]=size(input_data);
            % check the input patch_num value
            % if there is no value being input send out an error message
            % (4000 patches)
            if ~exist('patch_num','var')
                error("The number of the patches has not been set. Please double check.")
            else
                patch_num_perform=patch_num;
            end
            % check the input patch_size value
            % if there is no value being input set it to the default value
            if ~exist('patch_size','var')
                error("The size of the patches has not been set. Please double check.")
                % else, check the dimension of the input patch size
            else
                [row_input,col_input]=size(patch_size);
                % if the input is a single value, set the row, column, and channel
                % size to this input value
                if row_input==1 && col_input==1
                    row_patch=patch_size;
                    col_patch=patch_size;
                    chn_patch=patch_size;
                    % if the input is a vector, set the first value as the row size,
                    % second as the column size, and third as the channel size
                else
                    if row_input>col_input
                        row_patch=patch_size(1,1);
                        col_patch=patch_size(2,1);
                        chn_patch=patch_size(3,1);
                    else
                        row_patch=patch_size(1,1);
                        col_patch=patch_size(1,2);
                        chn_patch=patch_size(1,3);
                    end
                end
            end
            % Forming patches
            patched_data=zeros(row_patch,col_patch,chn_patch,patch_num);
            % retrieve input data size
            [row_input_data,col_input_data,chn_input_data]=size(input_data);
            % set up parameters for generating random index for row, column and
            % channel
            % row-wise parameters
            ind_row_min=1;
            ind_row_max=row_input_data;
            % column-wise parameters
            ind_col_min=1;
            ind_col_max=col_input_data;
            % channel-wise parameters
            ind_chn_min=1;
            ind_chn_max=chn_input_data;
            for i_patch=1:patch_num
                % generate random index range for row
                rand_row_ind=floor(ind_row_min+rand(1,1)*(ind_row_max-ind_row_min));
                if rand_row_ind > row_input_data-row_patch
                    start_ind_row=rand_row_ind-row_patch;
                    end_ind_row=rand_row_ind;
                else
                    start_ind_row=rand_row_ind;
                    end_ind_row=rand_row_ind+row_patch;
                end
                % generate random index range for column
                rand_col_ind=floor(ind_col_min+rand(1,1)*(ind_col_max-ind_col_min));
                if rand_col_ind > col_input_data-col_patch
                    start_ind_col=rand_col_ind-col_patch;
                    end_ind_col=rand_col_ind;
                else
                    start_ind_col=rand_col_ind;
                    end_ind_col=rand_col_ind+col_patch;
                end
                % generate random index range for channel
                rand_chn_ind=floor(ind_chn_min+rand(1,1)*(ind_chn_max-ind_chn_min));
                if rand_chn_ind > chn_input_data-col_patch
                    start_ind_chn=rand_chn_ind-chn_patch;
                    end_ind_chn=rand_chn_ind;
                else
                    start_ind_chn=rand_chn_ind;
                    end_ind_chn=rand_chn_ind+chn_patch;
                end
            end
            patched_data=gpuArray(patched_data);
        end 

    end
end