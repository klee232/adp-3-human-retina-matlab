% Created by Kuan-Min Lee
% Created date: Dec. 5th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This is the customized neural network layer which serves as the function
% of patch mean removal 

% Input Parameter:
% input: original image input into neural network layer (4D array)
% kernel_size: size of patch for calculation

% Output Parameter:
% output: processed array (4D array with size of 
% [num_slice, (kernel_size*kernel_size), (num_row*num_col)])

classdef mean_removal_layer < nnet.layer.Layer
    properties
        Kernel_size
    end
    
    methods

        %% Constructor for the layer
        function layer = mean_removal_layer(name,kernel_size)
            layer.Name = name;
            layer.Description = "mean_removal_lyr";
            layer.Kernel_size = kernel_size;
        end

        %% Forward pass: patch mean removal
        function output=predict(layer, input)
            kernel_size=layer.Kernel_size;
            %% grab out dimensional information
            [num_slice, num_row, num_column, num_batch, num_sample]=size(input);

            %% conduct patch mean removal
            % Create a kernel for computing the mean of a 3x3 patch
            kernel=ones(kernel_size,kernel_size)/(kernel_size*kernel_size);
            kernel=reshape(kernel,1,kernel_size,kernel_size);
            kernel=dlarray(kernel,'SSS');
            % reshape input
            reshaped_input=reshape(input,num_slice,num_row,num_column,num_batch*num_sample);
            reshaped_input=dlarray(reshaped_input,'SSSB');
            % generate mean patch
            mean_input=dlconv(reshaped_input,kernel,0,'Padding','same');
            % patch mean removal
            patch_mean_removal=reshaped_input-mean_input;

            %% reform outcome array
            output=extractdata(patch_mean_removal);
            output=reshape(output,num_slice,num_row,num_column,num_batch,num_sample);
            output=dlarray(output);

        end
       
    end
end