% Created by Kuan-Min Lee
% Created date: Dec. 15th, 2023
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This the customized 3D convolutional layer for building neural network

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

function layer=build_conv3D(input_feature,kernel_size,num_Filters,Weight_param,Bias_param,Stride,Padding)

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
           % if padding not exists, assign nothing
           if exist("Stride","var") && (ndims(Stride)<=3)
               s=Stride;
               % check if Padding exists
               if exist("Padding","var")
                   % if exists, and its type is same, assigned it to
                   % padding. Otherwise, do nothing
                   if Padding=="same" || ismatrix(Padding)
                       % retrieve dimensional information
                       [input_row,input_col,~]=size(input_feature);
                       % padding size for row
                       p_row=((Stride-1)*input_row-Stride+kernel_size)/2;
                       % padding size for column
                       p_col=((Stride-1)*input_col-Stride+kernel_size)/2;
                       % padding the input image
                       out_feat=padarray(input_feature,[p_row p_col],0,'both');
                   end
               end
               

                 
                
    
end