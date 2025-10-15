% Created by Kuan-Min Lee
% Created date: Jan. 29th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% I created tis customized convolutional layer class because I can't stand
% with MATLAB's stupid training network function

% Setup Parameter:
% gamma_val (optional): intended gamma value for batchnormalization
% beta_val (optional): intended beta value for batchnormalization
% Constant Parameter:
% e: small constant avoid the division of zero
% Input Parameter:
% input_feats: input feature maps (multi-dimensional array)
% Output Parameter
% output_feats: output feature maps (multi-dimensional array)

% needed modification

classdef Batchnormalization_layer < nnet.layer.Layer 
    properties(Constant)
        % Small constant to prevent division by zero.
        e = 1e-5;
    end
    % define the learnable paramters for convolutional 2D layer 
    % (in this case, the weights and bias)
    properties (Learnable)
        % Learnable parameters for convolutional 2D layer
        gamma
        beta
    end
    % define all the functions necessary for the layer
    methods
        % constructor function of the layer
        function layer = Batchnormalization_layer(name,chn_in,gamma_val,beta_val)
            % layer = Batchnormalization_layer to create a
            % batchnormlaization layer
            % set all input properties
            % String parts
            layer.Name=name; % name of current layer
            layer.Description="Customized Batchnormalization Layer"; % description of current layer
            % Parameter parts
            % if gamma value is not set, set it to default value 1
            if ~exist("gamma_val",'var')
                layer.gamma=ones(1,1,chn_in); 
            else
                layer.gamma=gamma_val; % size of the filter
            end
            % if beta value is not set, set it to default value 0
            if ~exist("beta_val",'var')
                layer.beta=0; 
            else
                layer.beta=beta_val;
            end
        end
        % forward function of the layer
        function feat_map = predict(layer,input_feats)
            % input: 
            % layer: 
            % in this case, it's the batchnormalization layer created
            % from the constructor function
            % input_feats: 
            % the feature maps being fed to the model
            % output:
            % feat_map: output feature maps
            % input_feats=gpuArray(input_feats);
            % calculate the mean and variance channel-wise
            mean_chn=mean(input_feats,[1 2]);
            var_chn=var(input_feats,0,[1 2]);
            % conduct calculation of batchnormalization
            x_hat=(input_feats-mean_chn)./sqrt(var_chn+layer.e);
            feat_map=layer.gamma*x_hat+layer.beta;
            % feat_map=gpuArray(feat_map);
        end
        % backward function of the layer
        function [dLdX,dLdr,dLdb] = backward(layer,input_feats,loss_feat)
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
            % dLdr: gradient with respect to the current gamma
            % dLdb: gradient with respect to the current beta
            % input_feats=gpuArray(input_feats);
            % loss_feat=gpuArray(loss_feat);
            % conduct calculation of dLdX
            mean_chn=mean(input_feats,[1 2]);
            var_chn=var(input_feats,0,[1 2]);
            [num_row,num_col,~]=size(input_feats);
            num_pix=num_row*num_col;
            % conduct calculation of dLdr (checked)
            x_hat=(input_feats-mean_chn)./sqrt(var_chn+layer.e);
            dLdr=sum(loss_feat.*x_hat,"all");
            % conduct calculation of dLdb (checked)
            dLdb=sum(loss_feat,"all");
            % conduct calculation of dLdX (Need to modify)
            % x_hat (checked)
            dLdx_hat=loss_feat*layer.gamma;
            dx_hatdx=1/(sqrt(var_chn+layer.e));
            x_hat_term=dLdx_hat.*dx_hatdx;
            % x_var (checked)
            dx_hatdvar=(-1/2)*(input_feats-mean_chn).*(var_chn+layer.e).^(-3/2);
            dvardx=2/num_pix*(input_feats-mean_chn);
            x_var_term=sum(dLdx_hat.*dx_hatdvar,[1 2]).*dvardx;
            % x_mean (checked)
            dx_hatdm=(-1)*(1/sqrt(var_chn+layer.e));
            dmdx=1/num_pix;
            x_m_term=sum(dLdx_hat,[1 2]).*dx_hatdm.*dmdx;
            dLdX=x_hat_term+x_var_term+x_m_term;
            % dLdX=gpuArray(dLdX);
            % dLdr=gpuArray(dLdr);
            % dLdb=gpuArray(dLdb);
        end
    end
end