% Created by Kuan-Min Lee
% Created date:: Jan. 9th, 2025

% Brief User Introduction:
% (revised from Morgan collab)
% This module utilize canny edge detection method to grab out the edge
% first and then fill the holes to conduct binarization

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% p2: struct to store the process information
% surface_deep_image: input 3D image with only surf and deep capillary layer (3D image array)
% mask: mask for segmentation layer (3D mask array)
% surf_thres: threshold for canny edge (surface capillary layer)
% deep_thres: threshold for canny edge (deep capillary layer)

% Output:
% p2: struct to store the process information
% binary_surf_deep_image: outcome of binarization image (binarized 3D image) 

% optimal setting:
% surf_thres=0.3; 
% deep_thres=0.3; 


function [p2,binary_surf_deep_image]=image_groundtruth_generator_binary_edge(p2,surface_deep_image,mask,surf_thres,deep_thres)
    disp("First Image Binarization IS RUNNING ...")
    
    %% partition the image
    mask_surf=zeros(size(mask));
    mask_deep=zeros(size(mask));
    mask_surf(mask>=2 & mask<=6)=1;
    mask_deep(mask==8)=1;
    surface_image=surface_deep_image.*mask_surf;
    deep_image=surface_deep_image.*mask_deep;


    %% conduct binarization for surface, deep and choroid layer image
    % surface capillary layer
    binary_surface_image=zeros(size(surface_image));
    num_surf_slice=size(surface_image,1);
    for i_surf_slice=1:num_surf_slice
        current_surf_slice=surface_image(i_surf_slice,:,:);
        current_surf_slice=squeeze(current_surf_slice);
        current_surf_slice_edge=edge(current_surf_slice,'canny',surf_thres);
        binary_surface_image(i_surf_slice,:,:)=imfill(current_surf_slice_edge,'holes');
    end

    % deep capillary layer
    binary_deep_image=zeros(size(deep_image));
    num_deep_slice=size(deep_image,1);
    for i_deep_slice=1:num_deep_slice
        current_deep_slice=surface_image(i_deep_slice,:,:);
        current_deep_slice=squeeze(current_deep_slice);
        current_deep_slice_edge=edge(current_deep_slice,'canny',deep_thres);
        binary_deep_image(i_deep_slice,:,:)=imfill(current_deep_slice_edge,'holes');
    end

    %% fuse two images
    binary_surf_deep_image=binary_surface_image+binary_deep_image;


    %% store outcome into struct
    p2.binary_img_rdhe_frangi_surf_deep=binary_surf_deep_image;
    disp("First Image Binarization COMPLETED.")
end