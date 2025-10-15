% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)

% Current Progression: Beta version


classdef ReLu_layer < nnet.layer.Layer 
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = ReLu_layer(name)
            % layer = Convolutional_2D_layer creates a convolutional 2D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="ReLu Activation function"; % description of current layer
        end
        % forward function of the layer
        function feat_map = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the acttivation layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % input_feats=gpuArray(input_feats);
            % filter out elements less than zeros
            feat_map=input_feats;
            feat_map(feat_map<=0)=0;
            % feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX] = backward(layer,input_feats,loss_feat)
            input_feats=gpuArray(input_feats);
            loss_feat=gpuArray(loss_feat);
            % input: 
            % layer: 
            % in this case, it's the ReLu layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % loss_feat:
            % the gradient of loss with respect to the input of the
            % previous layer
            % output:
            % dLdX: gradient with respect to the current input
            % input_feats=gpuArray(input_feats);
            [row_feat,col_feat,chn_feat]=size(input_feats);
            out_gradient=zeros(row_feat,col_feat,chn_feat);
            out_gradient(input_feats>0)=1;
            out_gradient=gpuArray(out_gradient);
            loss_feat=gpuArray(loss_feat);
            dLdX=loss_feat.*out_gradient;
            % dLdX=gpuArray(dLdX);
        end
    end
end