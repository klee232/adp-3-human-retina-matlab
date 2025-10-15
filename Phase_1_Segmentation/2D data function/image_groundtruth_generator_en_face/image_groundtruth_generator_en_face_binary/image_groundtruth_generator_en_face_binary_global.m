% Created by Kuan-Min Lee
% Created date:: Jan. 9th, 2024

% Brief User Introduction:
% (revised from Morgan collab)
% This module utilize a global thresholding method of creating a
% first-phase binarization result for the input 3D image (OCTA or OCT
% image)

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% p2: struct to store the process information
% surface_deep_image: OCTA image with only surface and deep capillary layer (3D image array)
% mask: mask for segmentation layer (3D mask array)
% surf_thres_ratio: threshold for binarization for surface capillary layer
% deep_thres_ratio: threshold for binarization for deep capillary layer

% Output:
% binary_surf_data_image: outcome of binarization image (binarized 3D image) 



function [binary_data_storage]=image_groundtruth_generator_en_face_binary_global(data_storage)
    disp("First Image Binarization IS RUNNING ...")
    
    %% conduct binarization for each file
    num_file=size(data_storage,1);
    binary_data_storage=zeros(size(data_storage));
    for i_file=1:num_file
        current_data_img=data_storage(i_file,:,:);
        current_data_img=squeeze(current_data_img);
        binary_data_storage(i_file,:,:)=imbinarize(current_data_img);
    end

    disp("First Image Binarization COMPLETED.")
    
end