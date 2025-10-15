% Created by Kuan-Min Lee
% Created date: May 15th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function will windowing the original image into 1/8 of its original
% size (passed testing)

% Setup Parameter:
% img_train: training dataset for partitioning
% gt_train: groundtruth dataset for partitioning
% Output:
% wind_img_train: windowed training image
% wind_gt_train: windowed groundtruth image

function [wind_img_train,wind_gt_train]=windowing_function(img_train,gt_train)
    % retrieve the size information
    [row_img,col_img,num_img]=size(img_train);
    % start partitioning
    wind_img_train=zeros(row_img/8,col_img/8,225,num_img,'uint8');
    wind_gt_train=zeros(row_img/8,col_img/8,225,num_img,'uint8');
    for i_img=1:num_img
        pointer=1;
        for i_row=1:row_img/16:((row_img-row_img/8)+1)
            for i_col=1:col_img/16:((col_img-col_img/8)+1)
                current_window_train=img_train(i_row:i_row+(row_img/8-1),i_col:i_col+(col_img/8-1),i_img);
                current_window_gt=gt_train(i_row:i_row+(row_img/8-1),i_col:i_col+(col_img/8-1),i_img);
                wind_img_train(:,:,pointer,i_img)=current_window_train;
                wind_gt_train(:,:,pointer,i_img)=current_window_gt;
                pointer=pointer+1;
            end
        end
    end
end