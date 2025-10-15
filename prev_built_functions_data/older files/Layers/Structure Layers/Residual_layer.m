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


classdef Residual_layer < nnet.layer.Layer 
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Residual_layer(name)
            % layer = Convolutional_3D_layer creates a convolutional 3D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Residual Layer"; % description of current layer
        end
        % forward function of the layer
        function out_feat = predict(layer,input_feats1,input_feats2)
            % input: 
            % layer: 
            % in this case, it's the convolutional 3D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            out_feat=input_feats1+input_feats2;
        end
        function [dLdY] = backward(layer,loss_feat_out,loss_feat_in)
            % input: 
            % layer: 
            % in this case, it's the convolutional 3D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            first_term=loss_feat_out;
            second_term=loss_feat_out.*loss_feat_in;
            dLdY=first_term+second_term;
        end
    end
end