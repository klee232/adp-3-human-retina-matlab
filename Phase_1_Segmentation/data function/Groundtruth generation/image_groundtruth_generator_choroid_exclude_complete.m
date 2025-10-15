% Created by Kuan-Min Lee
% Createed date: May 23rd, 2024

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% picture_obj_octa: picutre object struct for storing the images array from
% previous phase
% filtered_OCTA_img: region filtered octa image from previous phase (3D
% image cell array)
% filtered_seg_mask: region filtered segmentation mask (3D image cell
% array)

% Output:
% OCTA_img_binary: first phase segmentation image (3D image cell array)
% picture_obj_octa: picutre object struct for storing the images array 


function [OCTA_img_binary, picture_obj_octa]=image_groundtruth_generator_choroid_exclude_complete(picture_obj_octa,filtered_OCTA_img,filtered_seg_mask)
    
    %% partition the octa image into surface, deep, and choroid layer
    % create corresponding binary mask
    mask_choroid=ones(size(filtered_seg_mask));
    mask_choroid(filtered_seg_mask==14)=0;
    % partition the the octa image into three corresponding part
    OCTA_choroid_excluded=filtered_OCTA_img.*double(mask_choroid);


    %% conduct image denoising (dynamic histogram equalization)
    % initialize regional dynamic histogram equalization
    % surface and deep capillary (checked)
    region_size=[1,4,4];
    enhanced_filtered_OCTA_img_choroid_excluded=image_groundtruth_generator_denoise_rdhe_3d(OCTA_choroid_excluded,region_size);
  

    %% conduct image feature extraction (Frangi filter)
    % setup parameters
    % surface and deep capillary (checked)
    vessel_parameters = struct;
    vessel_parameters.DoG_scale_list = [1, 2, 4];
    vessel_parameters.alpha = 0.01;
    vessel_parameters.beta = 0.2;
    [picture_obj_octa,filtered_img_rdhe_frangi_choroid_excluded]=image_groundtruth_generator_feature_frangi_combined(picture_obj_octa,enhanced_filtered_OCTA_img_choroid_excluded,vessel_parameters);
   

    %% conduct first phase binarize image segmentation 
    % surface and deep capillary
    surf_sens_ratio=0.5; % filtered out 50% smallest pixels (tuned)
    deep_sens_ratio=0.5; % filtered out 50% smallest pixels (tuned)
    [picture_obj_octa,OCTA_img_choroid_excluded_binary]=image_groundtruth_generator_binary_local(picture_obj_octa,filtered_img_rdhe_frangi_choroid_excluded,filtered_seg_mask,surf_sens_ratio,deep_sens_ratio); % OCTA image
   

    %% save picture_obj
    OCTA_img_choroid_excluded_binary(find(OCTA_img_choroid_excluded_binary))=1;
    OCTA_img_binary=OCTA_img_choroid_excluded_binary;
    OCTA_img_binary(find(OCTA_img_binary))=1;
    picture_obj_octa.binary_img_rdhe_frangi=OCTA_img_binary;

end