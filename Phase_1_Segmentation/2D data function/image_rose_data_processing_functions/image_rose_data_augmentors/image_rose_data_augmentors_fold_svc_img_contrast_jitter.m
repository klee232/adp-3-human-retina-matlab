% Created by Kuan-Min Lee
% Created date: Dec. 8th, 2023
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to angle rotations (data
% augmentation) for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% data: training image that is intended to create data augmentation
% (multi-dimensional array)
% gt_data: groundtruth image that is intended to create data augmentation
% (multi-dimensional array)
% gamma_range: (array with range)
% beta_range: (array with range)

% Output Parameter
% augmentated_data: image rotated original octa image (multi-dimensional
% array)
% augmentated_gt_data: image rotated original octa image (multi-dimensional
% array)


function [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_contrast_jitter(data, gt_data, thick_gt_data, thin_gt_data, gamma_range, beta_range)
    %% get dimensional information
    num_row=size(data,1);
    num_col=size(data,2);
    num_sample=size(data,3);
    num_fold=size(data,4);


    %% create augmentated data storage variable
    augmentated_data=zeros(num_row,num_col, 2*num_sample, num_fold);
    augmentated_gt_data=zeros(num_row,num_col, 2*num_sample, num_fold);
    augmentated_thick_gt_data=zeros(num_row,num_col, 2*num_sample, num_fold);
    augmentated_thin_gt_data=zeros(num_row,num_col, 2*num_sample, num_fold);
    augmentated_data(:,:,1:num_sample,:)=data;
    augmentated_gt_data(:,:,1:num_sample,:)=gt_data;
    augmentated_thick_gt_data(:,:,1:num_sample,:)=thick_gt_data;
    augmentated_thin_gt_data(:,:,1:num_sample,:)=thin_gt_data;
    sample_ind=num_sample;


    %% conduct image elastic deformation (checked and updated)
    gamma=gamma_range(1)+diff(gamma_range)*rand;
    beta=beta_range(1)+diff(beta_range)*rand;
    for i_fold=1:num_fold
        for i_sample=1:num_sample
            current_data=data(:,:,i_sample,i_fold);
            current_gt_data=gt_data(:,:,i_sample,i_fold);
            current_thick_gt_data=thick_gt_data(:,:,i_sample,i_fold);
            current_thin_gt_data=thin_gt_data(:,:,i_sample,i_fold);
            current_data=squeeze(current_data);
            current_gt_data=squeeze(current_gt_data); 
            current_thick_gt_data=squeeze(current_thick_gt_data);
            current_thin_gt_data=squeeze(current_thin_gt_data);
            current_data_norm=current_data/255;
            current_data_norm_out=gamma*(current_data_norm-0.5)+0.5+beta;
            augmentated_data(:,:,sample_ind+i_sample,i_fold)=current_data_norm_out*255;
            augmentated_gt_data(:,:,sample_ind+i_sample,i_fold)=current_gt_data;
            augmentated_thick_gt_data(:,:,sample_ind+i_sample,i_fold)=current_thick_gt_data;
            augmentated_thin_gt_data(:,:,sample_ind+i_sample,i_fold)=current_thin_gt_data;
        end
    end
       
end