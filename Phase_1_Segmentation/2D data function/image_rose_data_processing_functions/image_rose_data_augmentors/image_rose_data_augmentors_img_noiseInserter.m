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

function [out_img]=image_rose_data_augmentors_img_noiseInserter(noise_sacle,image)
    %% Pre-check (checked and updated)
    % check if it's at least a 2D array type variable
    [row,col,~]=size(image);
    if (row<2) || (col<2)
        error("The input variable is not a 2D array. Please input another variable.")
    end


    %% conduct image noise insertion (checked and updated)
    num_aug=numel(noise_sacle);

    % create an array for storing the output image array
    [row_img,col_img,num_feature]=size(image);
    out_img=zeros(row_img,col_img,num_feature*num_aug);
    for i_feat=1:num_feature
        for i_aug=1:num_aug
            % Conduct image insertion
            current_image=image(:,:,i_feat);
            current_img_noise=current_image+noise_sacle(i_aug)*(rand(size(current_image))-0.5);
            % store the current image into the output array
            out_img(:,:,(num_aug*(i_aug-1)+i_aug))=current_img_noise;
        end
    end
   
       
end