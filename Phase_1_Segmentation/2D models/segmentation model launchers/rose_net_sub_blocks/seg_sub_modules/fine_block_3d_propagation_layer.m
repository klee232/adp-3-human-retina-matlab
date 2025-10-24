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
classdef fine_block_3d_propagation_layer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        Filter_size
    end

    properties(Learnable)
        Weight
    end

    methods

        %% Constructor for the layer
        function layer = fine_block_3d_propagation_layer(name, filter_size)
           layer.Name = name;
           layer.Description="propagation layer";
           % grab out confidence map generated from previous stage
           num_feat=filter_size*filter_size;
           weights=randn(1,1,1,num_feat);
           layer.Weight=weights;
           layer.Filter_size=filter_size;
        end

        %% Forward pass: channel multiplication
        function [output]=predict(layer,input)
            %% set up propagation coefficients
            propagation=input(:,:,:,2:end);
            image=input(:,:,:,1);
            weights=layer.Weight;
            norm_weights=exp(weights)./sum(exp(weights),"all");
            flatten_propagation=sum(propagation.*norm_weights,4);


            %% conduct propagation
            output=zeros(size(image));
            filter_size=layer.Filter_size;
            num_row=size(image,2);
            num_col=size(image,3);
            for i_row=1:num_row
                for i_col=1:num_col
                    start_row_ind=max((i_row-floor(filter_size/2)),1);
                    end_row_ind=min((i_row+floor(filter_size/2)),num_row);
                    start_col_ind=max((i_col-floor(filter_size/2)),1);
                    end_col_ind=min((i_col+floor(filter_size/2)),num_col);
                    current_region=image(:,start_row_ind:end_row_ind,start_col_ind:end_col_ind,:,:);
                    current_region_propagation=flatten_propagation(:,start_row_ind:end_row_ind,start_col_ind:end_col_ind,:,:);
                    current_region_conv=current_region.*current_region_propagation;
                    current_region_conv=sum(current_region_conv(:),'all');
                    output(:,i_row,i_col,:,:)=current_region_conv;
                end
            end

            output=dlarray(output,"SSSCB");

        end
       
    end
end
