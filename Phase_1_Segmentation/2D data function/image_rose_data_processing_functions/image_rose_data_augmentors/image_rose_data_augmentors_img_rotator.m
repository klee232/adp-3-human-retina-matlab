% Created by Kuan-Min Lee
% Created date: Dec. 8th, 2023
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to angle rotations (data
% augmentation) for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% num_aug: number of data augmentation (integer)
% image: training image that is intended to create data augmentation
% (multi-dimentional array)

% Output Parameter
% out_img: processed image array

function [out_img]=image_rose_data_augmentors_img_rotator(angles,image)
    %% Pre-check (checked and updated)
    % check if it's at least a 2D array type variable
    [row,col,~]=size(image);
    if (row<2) || (col<2)
        error("The input variable is not a 2D array. Please input another variable.")
    end


    %% conduct image rotation (checked and updated)
    num_ang=numel(angles);
    % create an array for storing the output image array
    [row_img,col_img,num_feat]=size(image);
    out_img=zeros(row_img,col_img,num_feat*num_ang);
    for i_ang=1:num_ang
        % Conduct image rotation
        current_img_rot=imrotate(image,angles(i_ang),"nearest","crop");
        % store the current image into the output array
        out_img(:,:,i_ang)=current_img_rot;
    end
   
       
end