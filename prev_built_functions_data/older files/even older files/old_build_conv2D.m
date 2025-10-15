% Created by Kuan-Min Lee
% Created date: Dec. 15th, 2023
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This the customized 2D convolutional layer for building neural network

% Input Parameter:
% input_feature (numerical array)
% kernel_size: size of the convolutional kernel
% num_Filters: number of filters (single variable)
% Weight_param (optional): if assigned, this arguement will be assigned to
% the kernel parameters
% Bias_param (optional): if assigned, this arguement will be assigned to
% bias parameter

% Output Parameter:
% output_feature (numerical array)

function output_feature=old_build_conv2D(input_feature,kernel_size,num_Filters,Weight_param,Bias_param,Stride,Padding)

    % check if Weight_param exists
    % if exists assign this parameter to weight
    if exist("Weight_param","var")
        W=Weight_param;
        % check if Bias_param exists
        % if exists assign this parameter to bias
        if exist("Bias_param","var")
           B=Bias_param;
           % check if Stride exists
            % if exists assign this parameter to Stride
            if exist("Stride","var") && (ndims(Stride)<=2)
                s=Stride;
                % check if Padding exists
                if exist("Padding","var")
                    % if exists, and its type is same, assigned it to
                    % padding
                    if Padding=="same" || ismatrix(Padding)
                       p=Padding;
                       layer = convolution2dLayer(kernel_size,num_Filters, ...
                                                  'Weights', W, ...
                                                  'Bias', B, ...
                                                  'Stride', s, ...
                                                  'Padding',p);
                    end
                 % if padding not exists, assign nothing
                 else
                    layer = convolution2dLayer(kernel_size,num_Filters, ...
                                               'Weights', W, ...
                                               'Bias', B, ...
                                               'Stride', s);
                end
            % if stride not exists, assign nothing
            else
               layer = convolution2dLayer(kernel_size,num_Filters, ...
                                          'Weights', W, ...
                                          'Bias', B);
            end
        % if bias not exists, assign random number to bias
        else
           layer = convolution2dLayer(kernel_size,num_Filters, ...
                                      'Weights', W, ...
                                      'BiasInitializer', @(sz) rand(sz) * 0.0001);
        end
    % if nothing exists, assign random number to weights and bias only
    else
       layer = convolution2dLayer(kernel_size,num_Filters, ...
                                 'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
                                 'BiasInitializer', @(sz) rand(sz) * 0.0001);
    end

    % output the convolutional feature maps
    output_feature=activations(layer,input_feature,1);
    
end