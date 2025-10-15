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

classdef pca_convolution_layer < nnet.layer.Layer
    properties(Learnable)
        Weight
        Bias
    end

    properties
        Kernel_size
        Num_filter
    end
    
    methods

        %% Constructor for the layer
        function layer = pca_convolution_layer(name,input, kernel_size_z, kernel_size_x, kernel_size_y, num_filter)
            %% setup non-learnable parameters
            layer.Name = name;
            layer.Description = "pca_convolution_lyr";
            layer.Kernel_size = [kernel_size_z kernel_size_x kernel_size_y];
            % for pca convolution, the number of filters can't exced
            % kernel_size*kernel_size
            if num_filter<=kernel_size_z*kernel_size_x*kernel_size_y
                layer.Num_filter = num_filter;
            else
                layer.Num_filter = kernel_size_z*kernel_size_x*kernel_size_y;
            end
            
            %% setup learnable parameters
            % pca weight setup
            % grab out dimensional information
            [input_dep, input_height,input_width,input_batch,input_sample]=size(input);
            % generate linear index for 3d input
            num_patch_dep=input_dep-kernel_size_z+1;
            num_patch_height=input_height-kernel_size_x+1;
            num_patch_width=input_width-kernel_size_y+1;
            dep_ind_lin=(1:kernel_size_z)'+(0:num_patch_dep-1);
            height_ind_lin=(1:kernel_size_x)'+(0:num_patch_height-1);
            width_ind_lin=(1:kernel_size_y)'+(0:num_patch_width-1);
            dep_ind_lin=reshape(dep_ind_lin,[],1,1);
            height_ind_lin=reshape(height_ind_lin,1,[],1);
            width_ind_lin=reshape(width_ind_lin,1,1,[]);
            linear_ind=bsxfun(@plus,...
                bsxfun(@plus, ...
                dep_ind_lin, (height_ind_lin-1)*input_dep),...
                (width_ind_lin-1)*input_dep*input_height);
            % form patches using linear index
            if input_batch>1 && input_sample>1
                patches_input=input(linear_ind,:,:);
            else
                patches_input=input(linear_ind);
            end
            % reshape into the desired form
            patches_input=reshape(patches_input,kernel_size_z*kernel_size_x*kernel_size_y,[]);
            % calculate the pca weights
            pca_coeff=pca(patches_input*transpose(patches_input));
            num_vector=size(pca_coeff,2);
            stop_ind=min([num_vector num_filter]);
            layer.Num_filter=stop_ind;
            eigen_vector=pca_coeff(:,1:stop_ind);
            layer.Weight=reshape(eigen_vector,kernel_size_z,kernel_size_x,kernel_size_y,[]);
            layer.Bias = 0;
        end

        %% Forward pass: convolution
        function output=predict(layer, input)
            %% grab out dimensional information
            num_slice=size(input,1);
            num_row=size(input,2);
            num_column=size(input,3);
            num_batch=size(input,4);
            num_sample=size(input,5);

            %% conduct convolution
            kernel=layer.Weight;
            kernel=dlarray(kernel,'SSSU');
            bias=layer.Bias;
            num_filter=layer.Num_filter;
            output=zeros(num_slice,num_row,num_column,num_batch*num_filter,num_sample);
            for i_sample=1:num_sample
                % grab out current image
                current_input=input(:,:,:,:,i_sample);
                current_input=dlarray(current_input,'SSSC');
                % grab out current filter
                % conduct convolution
                output_current_sample=dlconv(current_input, kernel, 0,"Padding","same");
                output(:,:,:,:,i_sample)=reshape(output_current_sample,[num_slice num_row num_column num_batch*num_filter]);
            end
            % add bias
            output=output+bias;
            output=dlarray(output);
        end
       
    end
end