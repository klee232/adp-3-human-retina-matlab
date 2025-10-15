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


classdef Logistic_layer < nnet.layer.Layer 
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Logistic_layer(name)
            % layer = Convolutional_2D_layer creates a convolutional 2D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Logistic Activation function"; % description of current layer
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
            input_feats=gpuArray(input_feats);
            % conduct calculation
            first_element=e.^(-input_feats);
            feat_map=1/(1+first_element);
            feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX] = backward(layer,feat_map,loss_feat)
            feat_map=gpuArray(feat_map);
            loss_feat=gpuArray(loss_feat);
            % input: 
            % layer: 
            % in this case, it's the activation layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % loss_feat:
            % the gradient of loss with respect to the input of the
            % previous layer
            % output:
            % dLdX: gradient with respect to the current input
            % conduct calculation
            current_grad=feat_map.*(1-feat_map);
            dLdX=loss_feat.*current_grad;
            dLdX=gpuArray(dLdX);
        end
    end
end