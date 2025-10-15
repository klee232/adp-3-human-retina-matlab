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

% Current Progression: beta version


classdef Transposeconvolutional_2D_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 2D layer
        kernel_size
        stride
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
        function layer = Transposeconvolutional_2D_layer(name,Kernel_size,input_chn)
            % layer = Convolutional_2D_layer creates a convolutional 2D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized 2D Transposed Convolution"; % description of current layer
            % Parameter parts
            layer.kernel_size=Kernel_size; % size of the filter
            % set up He initialization for kernel
            std=sqrt(2/(input_chn*Kernel_size*Kernel_size));
            std=[std 0;0 std];
            mu=[0 0];
            x=1:layer.kernel_size;
            y=1:layer.kernel_size;
            [x_scale,y_scale]=meshgrid(x,y);
            scales=[x_scale(:) y_scale(:)];
            values=mvncdf(scales,mu,std);
            layer.Weights=reshape(values,layer.kernel_size,layer.kernel_size); % Weights matrix
            layer.Bias=0; % Bias value
            layer.stride=2;
        end
        % forward function of the layer
        function feat_map = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the convolutional 2D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            [row_size,col_size,chn_size]=size(input_feats);
            out_row_size=2*row_size;
            out_col_size=2*col_size;
            out_chn_size=chn_size;
            input_feats=gpuArray(input_feats);
            feat_map=zeros(out_row_size,out_col_size,out_chn_size);
            feat_map=gpuArray(feat_map);
            % conduct convolution
            for i_feat=1:chn_size
                current_feat=input_feats(:,:,i_feat);
                % apply kronecker tensor product
                outcome=kron(current_feat,layer.Weights);
                if i_feat==1
                    feat_map=outcome;
                else
                    feat_map=cat(3,feat_map,outcome);
                end
            end
            % add bias
            feat_map=feat_map+layer.Bias;
            feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX,dLdW,dLdB] = backward(layer,input_feats,loss_feat)
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
            input_feats=gpuArray(input_feats);
            loss_feat=gpuArray(loss_feat);
            % calculate dLdW
            [row_kernel,col_kernel,chn_kernel]=size(layer.Weights);
            [row_in, col_in,~]=size(input_feats);
            [row_out,col_out,~]=size(loss_feat);
            dLdW=zeros(size(layer.Weights));
            for i_chn=1:chn_kernel
                for i_col=1:col_kernel
                    for i_row=1:row_kernel
                        start_row_ind=i_row;
                        end_row_ind=row_out;
                        start_col_ind=i_col;
                        end_col_ind=col_out;
                        current_region=loss_feat(start_row_ind:layer.stride:end_row_ind,start_col_ind:layer.stride:end_col_ind,i_chn:chn_kernel:end);
                        current_gradient=input_feats.*current_region;
                        dLdW(i_row,i_col,i_chn)=sum(current_gradient,"all");
                    end
                end
            end
            % calculate dLdX
            dLdX=convolution_with_stride(loss_feat,layer.Weights,layer.stride);
            % calculate dLdB
            dLdB=sum(loss_feat,"all");
            dLdW=gpuArray(dLdW);
            dLdX=gpuArray(dLdX);
            dLdB=gpuArray(dLdB);
        end
    end
end