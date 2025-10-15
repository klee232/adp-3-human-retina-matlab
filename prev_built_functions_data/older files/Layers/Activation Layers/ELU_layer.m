% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter
% alpha: parameter used for calculation (double)
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)

% Current Progression: Beta version


classdef ELU_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 2D layer
        alpha
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = ELU_layer(name)
            % layer = Convolutional_2D_layer creates a convolutional 2D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Exponential Linear Activation function"; % description of current layer
            layer.alpha=1;
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
            % conduct calculation
            input_feats=gpuArray(input_feats);
            feat_map=input_feats;
            feat_map(feat_map<=0)=layer.alpha*(e.^(input_feats)-1);
            feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX] = backward(layer,input_feats,loss_feat)
            input_feats=gpuArray(input_feats);
            loss_feat=gpuArray(loss_feat);
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
            % conduct calculation
            input_feats=gpuArray(input_feats);
            temp_input=input_feats;
            temp_input(temp_input<=0)=layer.alpha*e.^(input_feats);
            dLdX=loss_feat.*temp_input;
            dLdX=gpuArray(dLdX);
        end
    end
end