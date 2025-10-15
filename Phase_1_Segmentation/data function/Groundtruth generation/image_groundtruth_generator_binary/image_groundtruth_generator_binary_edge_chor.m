% Created by Kuan-Min Lee
% Created date:: May 23rd, 2024

% Brief User Introduction:
% (revised from Morgan collab)
% This module utilize a global thresholding method of creating a
% first-phase binarization result for the input 3D image (OCTA or OCT
% image)

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% p2: struct to store the process information
% original_image: input 3D image (3D image array)
% mask_binary: mask for segmentation layer (3D mask array)

% Output:
% p2: struct to store the process information
% binary_image: outcome of binarization image (binarized 3D image) 

% optimal setting:
% choroid_thres_ratio=0.85; % filtered out 5% smallest pixels (tuned)

function [p2,binary_choroid_image]=image_groundtruth_generator_binary_edge_chor(p2,choroid_image)
    disp("First Image Binarization IS RUNNING ...")
  
   %% conduct binarization for  choroid layer image
    % surface capillary layer
    binary_choroid_image=zeros(size(choroid_image));
    num_choroid_slice=size(choroid_image,1);
    for i_choroid_slice=1:num_choroid_slice
        current_choroid_slice=choroid_image(i_choroid_slice,:,:);
        current_choroid_slice=squeeze(current_choroid_slice);
        current_choroid_slice_edge=edge(current_choroid_slice,'canny',0.55);
        binary_choroid_image(i_choroid_slice,:,:)=imfill(current_choroid_slice_edge,'holes');
    end
    

    %% store outcome into struct
    p2.binary_img_rdhe_frangi_choroid=binary_choroid_image;
    disp("First Image Binarization COMPLETED.")
end