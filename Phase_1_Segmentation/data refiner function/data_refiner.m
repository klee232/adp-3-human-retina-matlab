% Created by Kuan-Min Lee
% Createed date: Jan. 15th, 2025

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% octa_gt_storage: groundtruth variable generated from the previous
% phase

% Output:
% denoised_refined_octa_gt_storage: denoised groundtruth outcome


function [denoised_refined_octa_gt_storage]=data_refiner(octa_gt_storage)

    %% conduct manual denoising user interface
    [denoised_refined_octa_gt_storage]=image_groundtruth_refiner_denoise(octa_gt_storage);

    
    %% save the processed data inside the folder
    save("~/data/klee232/processed_data/refined_pad_octa_data_frangi.mat","denoised_refined_octa_gt_storage",'-v7.3');


end