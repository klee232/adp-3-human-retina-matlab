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
% surface_deep_image: OCTA image with only surface and deep capillary layer (3D image array)
% mask: mask for segmentation layer (3D mask array)
% surf_thres_ratio: threshold for binarization for surface capillary layer
% deep_thres_ratio: threshold for binarization for deep capillary layer

% Output:
% p2: struct to store the process information
% binary_surf_deep_image: outcome of binarization image for surf and deep capillary layer(binarized 3D image) 

% optimal setting:
% surf_thres_ratio=0.93; % filtered out 93% smallest pixels (tuned)
% deep_thres_ratio=0.93; % filtered out 93% smallest pixels (tuned)


function [p2,binary_surf_deep_image]=image_groundtruth_generator_binary(p2,surface_deep_image,mask,surf_thres_ratio,deep_thres_ratio)
    disp("First Image Binarization IS RUNNING ...")
    
    %% partition the image
    mask_surf=zeros(size(mask));
    mask_deep=zeros(size(mask));
    mask_surf(mask>=2 & mask<=6)=1;
    mask_deep(mask==8)=1;
    surface_image=surface_deep_image.*mask_surf;
    deep_image=surface_deep_image.*mask_deep;


    %% conduct binarization for surface, deep and choroid layer image
    % surface layer
    flat_surface_image=surface_image(:);
    flat_surface_image=flat_surface_image(flat_surface_image>0);
    sorted_surf_array=sort(flat_surface_image,'descend');
    surf_thres_ind=floor((1-surf_thres_ratio)*length(flat_surface_image(:)));
    surf_thres=sorted_surf_array(surf_thres_ind);
    binary_surface_image=surface_image;
    binary_surface_image(surface_image>=surf_thres)=1;
    binary_surface_image(surface_image<surf_thres)=0;

    % deep leayer
    flat_deep_image=deep_image(:);
    flat_deep_image=flat_deep_image(flat_deep_image>0);
    sorted_deep_array=sort(flat_deep_image,'descend');
    deep_thres_ind=floor((1-deep_thres_ratio)*length(flat_deep_image(:)));
    deep_thres=sorted_deep_array(deep_thres_ind);
    binary_deep_image=deep_image;
    binary_deep_image(deep_image>deep_thres)=1;
    binary_deep_image(deep_image<=deep_thres)=0;
    

    %% fuse two images
    binary_surf_deep_image=binary_surface_image+binary_deep_image;


    %% store outcome into struct
    p2.binary_img_rdhe_frangi_surf_deep=binary_surf_deep_image;
    disp("First Image Binarization COMPLETED.")
end