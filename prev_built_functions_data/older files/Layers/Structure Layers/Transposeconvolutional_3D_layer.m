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

% Current Progression: Incomplete


classdef Transposeconvolutional_3D_layer < nnet.layer.Layer 
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
        function layer = Transposeconvolutional_3D_layer(name,concate_dim)
            % layer = Convolutional_3D_layer creates a convolutional 3D
            % layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized 3D Convolution"; % description of current layer
            % Parameter parts
            layer.Concate_dim=concate_dim; % size of the filter
            layer.Weights=randn(kernel_size,kernel_size,kernel_size); % Weights matrix
            layer.Bias=0; % Bias value
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
            out_chn_size=2*chn_size;
            input_feats=gpuArray(input_feats);
            feat_map=zeros(out_row_size,out_col_size,out_chn_size);
            feat_map=gpuArray(feat_map);
            % conduct convolution
            for i_feat=1:chn_size
                for i_col=1:col_size
                    for i_row=1:row_size
                        % grab out current feature 
                        current_feat=input_feats(i_row,i_col,i_feat);
                        outcome=current_feat*layer.Weights;
                        % store current result into assigned output feature
                        % map locations
                        feat_map(1+layer.stride*(i_row-1):1+layer.stride*(i_row-1)+(layer.kernel_size-1),...
                                 1+layer.stride*(i_col-1):1+layer.stride*(i_col-1)+(layer.kernel_size-1),...
                                 1+layer.stride*(i_feat-1):1+layer.stride*(i_feat-1)+(layer.kernel_size-1))=outcome;
                    end
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
            % in this case, it's the convolutional 3D layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % create matrix for loss gradient with respectt to filter
            % parameters
            input_feats=gpuArray(input_feats);
            loss_feat=gpuArray(loss_feat);
            [row_feat,col_feat,chn_feat]=size(input_feats);
            dLdW=convn(input_feats,loss_feat,'same');
            rotated_weight=rot90(layer.Weights,2); % rotate the weight in the first dimension
            rotated_weight=imrotate3(rotated_weight,180,[0 0 1]); % rotate the weight in the 3rd dimension
            dLdX=convn(loss_feat,rotated_weight,'same');
            dLdB=loss_feat*ones(row_feat,col_feat,chn_feat);
            dLdW=gpuArray(dLdW);
            dLdX=gpuArray(dLdX);
            dLdB=gpuArray(dLdB);
        end
    end
end