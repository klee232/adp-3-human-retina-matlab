% Created by Kuan-Min Lee
% Created date: Dec. 8th, 2023
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to angle rotations (data
% augmentation) for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% data: training image that is intended to create data augmentation
% (multi-dimentional array)
% gt_data: groundtruth image that is intended to create data augmentation
% (multi-dimentional array)
% num_aug: 

% Output Parameter
% out_img: processed image array

function [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_rotator(data, gt_data, thick_gt_data, thin_gt_data, num_aug)
    %% get dimensional information
    num_row=size(data,1);
    num_col=size(data,2);
    num_sample=size(data,3);
    num_fold=size(data,4);


    %% create augmentated data storage variable
    augmentated_data=zeros(num_row,num_col, num_sample*(num_aug+1), num_fold);
    augmentated_gt_data=zeros(num_row,num_col, num_sample*(num_aug+1), num_fold);
    augmentated_thick_gt_data=zeros(num_row,num_col, num_sample*(num_aug+1), num_fold);
    augmentated_thin_gt_data=zeros(num_row,num_col, num_sample*(num_aug+1), num_fold);
    augmentated_data(:,:,1:num_sample,:)=data;
    augmentated_gt_data(:,:,1:num_sample,:)=gt_data;
    augmentated_thick_gt_data(:,:,1:num_sample,:)=thick_gt_data;
    augmentated_thin_gt_data(:,:,1:num_sample,:)=thin_gt_data;
    sample_ind=num_sample;

 
    %% conduct data augmentation 
    for i_fold=1:num_fold
        current_sample_ind=sample_ind+1;
        for i_sample=1:num_sample
            current_data=data(:,:,i_sample,i_fold);
            current_gt_data=gt_data(:,:,i_sample,i_fold);
            current_thick_gt_data=thick_gt_data(:,:,i_sample,i_fold);
            current_thin_gt_data=thin_gt_data(:,:,i_sample,i_fold);
            current_data=squeeze(current_data);
            current_gt_data=squeeze(current_gt_data);
            current_thick_gt_data=squeeze(current_thick_gt_data);
            current_thin_gt_data=squeeze(current_thin_gt_data);

            for i_aug=1:num_aug
                aug_ang=10*randn;
                current_aug_data=imrotate(current_data,aug_ang,'bilinear','crop');
                current_aug_gt_data=imrotate(current_gt_data,aug_ang,'bilinear','crop');
                current_aug_thick_gt_data=imrotate(current_thick_gt_data,aug_ang,'bilinear','crop');
                current_aug_thin_gt_data=imrotate(current_thin_gt_data,aug_ang,'bilinear','crop');
                augmentated_data(:,:,current_sample_ind,i_fold)=current_aug_data;
                augmentated_gt_data(:,:,current_sample_ind,i_fold)=current_aug_gt_data;
                augmentated_thick_gt_data(:,:,current_sample_ind,i_fold)=current_aug_thick_gt_data;
                augmentated_thin_gt_data(:,:,current_sample_ind,i_fold)=current_aug_thin_gt_data;
                current_sample_ind=current_sample_ind+1;
            end

        end
    end

       
end