% Created by Kuan-Min Lee
% Created date: Jan. 8th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This the customized 2D convolutional layer for building neural network

classdef conv2DLayer < nnet.layer.Layer ...
        & nnet.layer.Acceleratable

    % Define Layer Learnable Parameters
    properties (Learnable)
        Weights
        Bias
    end

    % Creates Relative Functions for the Layer
    methods

        % create constructor function
        function layer=conv2DLayer(args)
            arguments
                args.Name="";
            end

            % set layer name
            layer.Name=args.Name;

            % set layer descriptioon 
            layer.Description="Conv2D";
        end

        % create initialization function (for learnable parameters)
        function layer=initialize(layer,num_filter,kernel_size)
            % Check if the learnable parameters have already been
            % initialized
            % If yes, skip the initialization
            if ~isempty(layer.Weights) & ~isempty(layer.Bias)
                return
            end

            % Initialize Weights
            layer.Weights=rand(num_filter,kernel_size,kernel_size);

            % Initialize Bias
            layer.Bias=zeros(num_filter,1);
        end

        % create forward function
        function out_feat=conv2D_forward(layer, input_feat,pad_opt)
            % Initialize Weight and Bias
            w=layer.Weights;
            b=layer.Bias;

            % Retrieve Input Image Dimensional Information
            num_dims=ndims(input_feat); % number of dimensions
            % For cases of 2D images (array format: (num_imgs, rows cols)
            if num_dims==3
                [num_imgs,rows,cols]=size(input_feat);
            % For cases of 3D images (array format: (num_imgs, rows, cols,
            % chns)
            else
                [num_imgs,rows,cols,chns]=size(input_feat);
            end

            % Conduct Convolution
            % Check if the paddig is needed
            % If so, 
            

        end


    end
end