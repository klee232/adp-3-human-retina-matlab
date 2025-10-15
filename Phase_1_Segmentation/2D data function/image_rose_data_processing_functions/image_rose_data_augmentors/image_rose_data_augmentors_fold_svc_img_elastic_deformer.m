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
% alpha: (variable)
% beta: (variable)

% Output Parameter
% augmentated_data: image rotated original octa image (multi-dimensional
% array)
% augmentated_gt_data: image rotated original octa image (multi-dimensional
% array)


function [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_elastic_deformer(data, gt_data, thick_gt_data, thin_gt_data, alpha, sigma)
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


    %% create random displacement fields
    u0=randn(num_row,num_col);
    v0=randn(num_row,num_col);

    % smooth with gaussian
    gauss_filt=fspecial('gaussian', 2*round(3*sigma)+1, sigma);
    u=imfilter(u0,gauss_filt, 'replicate');
    v=imfilter(v0,gauss_filt, 'replicate');

    % scale the transformation
    u=alpha*u;
    v=alpha*v;

    % create spatial displacement field
    [col_map, row_map]=meshgrid(1:num_col,1:num_row);
    col_map=col_map+u;
    row_map=row_map+v;


    %% conduct image elastic deformation (checked and updated)
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
            augmentated_data(:,:,sample_ind+i_sample,i_fold)=interp2(double(current_data), col_map, row_map, 'linear', 0);
            augmentated_gt_data(:,:,sample_ind+i_sample,i_fold)=interp2(double(current_gt_data), col_map, row_map, 'nearest', 0);
            augmentated_thick_gt_data(:,:,sample_ind+i_sample,i_fold)=interp2(double(current_thick_gt_data), col_map, row_map, 'linear', 0);
            augmentated_thin_gt_data(:,:,sample_ind+i_sample,i_fold)=interp2(double(current_thin_gt_data), col_map, row_map, 'nearest', 0);
        end
    end
       
end