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


classdef Convolutional_3D_layer < nnet.layer.Layer 
    properties
    % define the unlearnable parameters for convolutional 3D layer
        kernel_size
    end
    % define the learnable paramters for convolutional 2D layer 
    % (in this case, the weights and bias)
    properties (Learnable)
        % Learnable parameters for convolutional 3D layer
        Weights
        Bias
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Convolutional_3D_layer(name,kernel_size)
            % layer = Convolutional_3D_layer creates a convolutional 3D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized 3D Convolution"; % description of current layer
            % Parameter parts
            layer.kernel_size=kernel_size; % size of the filter
            layer.Weights=randn(kernel_size,kernel_size,kernel_size); % Weights matrix
            layer.Bias=0; % Bias value
        end
        % forward function of the layer
        function feat_map = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the convolutional 3D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            feat_map=convn(input_feats,layer.Weights,'same');
            feat_map=feat_map+layer.Bias;
        end
        % backward function of the layer
        function [dLdX,dLdW,dLdB] = backward(layer,input_feats,loss_feat)
            % input: 
            % layer: 
            % in this case, it's the convolutional 3D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % create matrix for loss gradient with respectt to filter
            % parameters
            [row_feat,col_feat,chn_feat]=size(input_feats);
            dLdW=convn(input_feats,loss_feat,'same');
            rotated_weight=rot90(layer.Weights,2); % rotate the weight in the first dimension
            rotated_weight=imrotate3(rotated_weight,180,[0 0 1]); % rotate the weight in the 3rd dimension
            dLdX=convn(loss_feat,rotated_weight,'same');
            dLdB=loss_feat*ones(row_feat,col_feat,chn_feat);
        end
    end
end