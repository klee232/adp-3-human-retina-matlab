% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:
% kernel_size (optional): intended kernel size of max pooling
% sttride (optional): intended stride for kernel movement
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)


classdef MaxPooling2D_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 2D layer
        filt_size
        stride
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = MaxPooling2D_layer(name,kernel_size,Stride)
            % layer = MaxPooling2D_layer creates a maxpooling layer with
            % kernel_size
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized Batchnormalization Layer"; % description of current layer
            % Parameter parts
            % if kernel size value is not set, set it to default value 1
            if ~exist("kernel_size",'var')
                layer.filt_size=2; 
            else
                layer.filt_size=kernel_size; % size of the filter
            end
            % if stride value is not set, set it to default value 1
            if ~exist("stride",'var')
                layer.stride=kernel_size; 
            else
                layer.stride=Stride; % size of the filter
            end
        end
        % forward function of the layer
        function [feat_map,max_mask] = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the max pooling layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % conduct max pooling calculation
            % create output feature map storage
            input_feats=gpuArray(input_feats);
            [row_input,col_input,chn_input]=size(input_feats);
            row_feat=((row_input-layer.filt_size)/layer.stride)+1;
            col_feat=((col_input-layer.filt_size)/layer.stride)+1;
            chn_feat=chn_input;
            feat_map=zeros(row_feat,col_feat,chn_feat);
            max_mask=zeros(row_input,col_input,chn_input);
            feat_map=gpuArray(feat_map);
            % conduct max pooling 
            for i_chn=1:chn_feat
                % Generate max pooling outcome
                reshaped_input = im2col(input_feats(:,:,i_chn), [layer.filt_size layer.filt_size],"distinct");
                reshaped_input=reshape(reshaped_input,layer.filt_size,layer.filt_size,[]);
                [reshaped_maxouts,~] = max(reshaped_input,[],[1 2],"linear");
                maxouts=col2im(reshaped_maxouts,[1 1], [row_feat col_feat]);
                feat_map(:,:,i_chn)=maxouts;
                % Generate max masks
                current_max_mask=find_all_max_indices(input_feats(:,:,i_chn),layer.filt_size);
                max_mask(:,:,i_chn)=current_max_mask;
            end
            feat_map=gpuArray(feat_map);
            max_mask=gpuArray(max_mask);
        end
        % backward function of the layer
        function [dLdx] = backward(layer,max_mask,loss_feat)
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
            % dLdx: gradient with respect to the current input
            max_mask=gpuArray(max_mask);
            loss_feat=gpuArray(loss_feat);
            % create output variable storage
            [row_in,col_in,chn_in]=size(max_mask);
            dLdx=zeros(row_in,col_in,chn_in);
            for i_chn=1:chn_in
                % conduct reshaping
                current_loss=loss_feat(:,:,i_chn);
                reshpaed_current_loss=im2col(current_loss,[1 1],"distinct");
                current_gradient=max_mask(:,:,i_chn);
                reshaped_current_gradient = im2col(current_gradient, [layer.filt_size layer.filt_size],"distinct");
                % conduct multiplication
                reshaped_grad_wrt_input=reshaped_current_gradient.*reshpaed_current_loss;
                % reshape back to original input size
                current_grad_wrt_input = col2im(reshaped_grad_wrt_input, [layer.filt_size layer.filt_size], [row_in col_in], "distinct");
                % store it into dLdx matrix
                dLdx(:,:,i_chn)=current_grad_wrt_input;
            end
            dLdx=gpuArray(dLdx);
        end
    end
end