% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter
% kernel_size: size of the convolutional kernel (integer)
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)

% Current Progression: Beta version


classdef Convolutional_2D_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 2D layer
        num_filter
        kernel_size
        input_chn
        option
    end
    % define the learnable paramters for convolutional 2D layer 
    % (in this case, the weights and bias)
    properties (Learnable)
        % Learnable parameters for convolutional 2D layer
        Weights
        Bias
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Convolutional_2D_layer(name,Num_filter,Kernel_size,Input_chn,option)
            % layer = Convolutional_2D_layer creates a convolutional 2D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized 2D Convolution"; % description of current layer
            % Parameter parts
            layer.num_filter=Num_filter; % number of kernel for convolution
            layer.kernel_size=Kernel_size; % size of the filter
            layer.input_chn=Input_chn; % number of channel fed into the convolution
            layer.option=option; % convolutional option
            % set up He initialization for kernel
            std=sqrt(2/(layer.num_filter*layer.input_chn*layer.kernel_size));
            values=randn(layer.kernel_size,layer.kernel_size,layer.input_chn,layer.num_filter)*std;
            layer.Weights=values; % Weights matrix
            layer.Bias=0; % Bias value
        end
        % forward function of the layer
        function feat_map = predict(layer,Input_feats)
            % input: 
            % layer: 
            % in this case, it's the convolutional 2D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % Input_feats=gpuArray(Input_feats);
            [row_in,col_in,num_in]=size(Input_feats);
            % check option value
            if layer.option=="valid" || layer.option=="Valid"
                input_feats=Input_feats;
            elseif layer.option=="same" || layer.option=="Same"
                pad_size=(layer.kernel_size-1)/2;
                input_feats=padarray(Input_feats,[pad_size pad_size]);
            % default setting as valid
            else
                input_feats=Input_feats;
            end
            % conduct convolution
            % grab out the number of channel input into convolution
            stride_chn=layer.input_chn;
            num_filters=layer.num_filter;
            % if this is a 3D convolution
            if stride_chn~=1
                % partition the input image into slices for feeding into
                % convolution
                feat_map=zeros(row_in,col_in,num_in*num_filters/stride_chn);
                feat_map=gpuArray(feat_map);
                for i_filter=1:num_filters
                    current_w=layer.Weights(:,:,:,i_filter);
                    % current_w=gpuArray(current_w);
                    temp_conv=0;
                    % temp_conv=gpuArray(temp_conv);
                    for i_kern_chn=1:stride_chn
                        temp_conv=temp_conv+convn(input_feats(:,:,i_kern_chn:stride_chn:end),current_w(:,:,i_kern_chn,:),'valid');
                    end
                    feat_map(:,:,i_filter:num_filters:end)=temp_conv;
                end
            % if this is a 2D convolution
            else
                chn_out=num_in*num_filters;
                feat_map=zeros(row_in,col_in,chn_out);
                % feat_map=gpuArray(feat_map);
                for i_filter=1:num_filters
                    current_w=layer.Weights(:,:,:,i_filter);
                    % current_w=gpuArray(current_w);
                    feat_map(:,:,i_filter:num_filters:end)=convn(input_feats,current_w,'valid');
                end
            end
            feat_map=feat_map+layer.Bias;
            % feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX,dLdW,dLdB] = backward(layer,Input_feats,loss_feat)
            % input: 
            % layer: 
            % in this case, it's the convolutional 2D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % loss_feat:
            % the gradient of loss with respect to the input of the
            % previous layer
            % output:
            % dLdX: gradient with respect to the current input
            % dLdW: gradient with respect to the current weights
            % dLdB: gradient with respect to the current bias
            % calculate the gradient with respect to weights
            % check option value
            if layer.option=="valid" || layer.option=="Valid"
                input_feats=double(Input_feats);
            elseif layer.option=="same" || layer.option=="Same"
                pad_size=(layer.kernel_size-1)/2;
                input_feats=padarray(double(Input_feats),[pad_size pad_size]);
            % default setting as valid
            else
                input_feats=double(Input_feats);
            end
            % calculate dLdW (checked)
            chn_input=layer.input_chn;
            num_filters=layer.num_filter;
            dLdW=zeros(size(layer.Weights));
            % 3D convolution cases
            if chn_input~=1
                for i_filt=1:num_filters
                    for i_col=1:col_kernel
                        for i_row=1:row_kernel
                            [row_size,col_size,~]=size(loss_feat);
                            % grab out relative loss features
                            current_loss=loss_feat(:,:,:,i_filt:num_filters:end);
                            % grab out relative input features
                            current_region=input_feats(i_row:row_size+(i_row-1),i_col:col_size+(i_col-1),:,:);
                            current_gradient=double(current_loss).*current_region;
                            dLdW(i_row,i_col,:,i_filt)=sum(current_gradient,[1 2]);
                        end
                    end
                end
            % 2D convolution cases
            else
                for i_filt=1:num_filters
                    for i_col=1:col_kernel
                        for i_row=1:row_kernel
                            [row_size,col_size,~]=size(loss_feat);
                            % grab out relative loss features
                            current_loss=loss_feat(:,:,:,i_filt:num_filters:end);
                            % grab out relative input features
                            current_region=input_feats(i_row:row_size+(i_row-1),i_col:col_size+(i_col-1),:,:);
                            current_gradient=double(current_loss).*current_region;
                            dLdW(i_row,i_col,:,i_filt)=sum(current_gradient,[1 2]);
                        end
                    end
                end
            end
            % Compute gradients with respect to input dLdX
            rotated_weight=rot90(layer.Weights,2); % rotate the weight in the first dimension
            dLdX=zeros(size(input_feats));
            [~,~,~,num_data]=size(input_feats);
            for i_data=1:num_data
                current_loss=loss_feat(:,:,:,i_data:num_filters:end);
                dLdX(:,:,:,i_data)=convn(current_loss,rotated_weight,'full');
            end
            dLdX=dLdX(1+pad_size:end-pad_size,1+pad_size:end-pad_size,:);
            % Compute gradient with respect to bias dLdB
            dLdB=sum(loss_feat,'all');
        end
    end
end