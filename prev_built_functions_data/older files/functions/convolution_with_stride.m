% Created by Kuan-Min Lee and ChatGpt
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

function output = convolution_with_stride(input, kernel, stride)
  % Get dimensions of input and kernel
    [row_in, col_in, chn_in] = size(input);
    [kernel_row, kernel_col, ~] = size(kernel);
    
    % Calculate output dimensions
    row_out = floor((row_in - kernel_row) / stride) + 1;
    col_out = floor((col_in - kernel_col) / stride) + 1;
    
    % Initialize output
    output = zeros(row_out, col_out, chn_in);
    
    % Perform convolution for each depth slice
    for i_chn=1:chn_in
        % Reshape the padded array into a 3D array of size (2, 2, numMasks)
        reshaped_input = reshape(input(:,:,i_chn), kernel_row, kernel_col, []);
        reshaped_output=reshaped_input.*kernel;
        reshaped_convout = sum(reshaped_output,[1 2]);
        convouts=reshape(reshaped_convout,row_out,col_out);
        output(:,:,i_chn)=convouts;
    end
end