% Created by Kuan-Min Lee
% Created date:: Jan. 9th, 2025

% Brief User Introduction:
% This module utilize a adaptive local thresholding method of creating a
% first-phase binarization result for the choroid layer OCTA image

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% p2: struct to store the process information
% choroid_image: OCTA image with only choroid layer (3D image array)
% choroid_thres_ratio: threshold for binarization

% Output:
% p2: struct to store the process information
% binary_choroid_image: outcome of binarization image for choroid layer (binarized 3D image) 

% optimal setting:
% choroid_thres_ratio=0.85; % filtered out 5% smallest pixels (tuned)

function [p2,binary_choroid_image]=image_groundtruth_generator_en_face_binary_local_chor(p2,choroid_image,choroid_thres_ratio)
    disp("First Image Binarization IS RUNNING ...")
  
   %% conduct binarization for  choroid layer image
    % surface capillary layer
    binary_choroid_image=zeros(size(choroid_image));
    num_choroid_slice=size(choroid_image,1);
    for i_choroid_slice=1:num_choroid_slice
        current_choroid_slice=choroid_image(i_choroid_slice,:,:);
        current_choroid_slice=squeeze(current_choroid_slice);
        current_choroid_slice_binary=adaptthresh(current_choroid_slice,choroid_thres_ratio);
        binary_choroid_image(i_choroid_slice,:,:)=imbinarize(current_choroid_slice,current_choroid_slice_binary);
    end
    

    %% store outcome into struct
    p2.binary_img_rdhe_frangi_choroid=binary_choroid_image;
    disp("First Image Binarization COMPLETED.")
end