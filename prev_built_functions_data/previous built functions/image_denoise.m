% Created by Kuan-Min Lee
% Createed date:: May 23rd, 2024

% Brief User Introduction:
% (copied from Morgan collab)
% This module utilize a 3D median filter with size of [11 11 11] and a
% global threshold binarizer with threshold of 98.7% highest value to
% conduct image denoising

% Input parameter:
% p2: struct to store the process information
% II: input 3D image
% opt: operation for denoising

% Output:
% p2: struct to store the process information
% binary_II: outcome of denoise (denoised 3D image) 

function [p2,denoised_image]=image_denoise(p2,original_image,mask_seg)
    disp("Image Denoise IS RUNNING ...")

    % partition the sample into two pieces (shallower and choroid layer)
    % create corresponding binary mask
    mask_binary=mask_seg;
    mask_binary(mask_binary<14)=0;
    mask_binary(mask_binary==14)=1;
    mask_binary_rev=1-mask_binary;
    % partitioning sample into two layers
    shallow_image=original_image.*double(mask_binary_rev);
    choroid_image=original_image.*double(mask_binary);

    % conduct binarization for shallow and choroid layer image
    thres_shallow=0.4;
    binary_shallow_image=shallow_image;
    binary_shallow_image(binary_shallow_image>thres_shallow)=1;
    binary_shallow_image(binary_shallow_image<=thres_shallow)=0;
    thres_choroid=0.37;
    binary_choroid_image=choroid_image;
    binary_choroid_image(binary_choroid_image>thres_choroid)=1;
    binary_choroid_image(binary_choroid_image<=thres_choroid)=0;
    
    % [num_slice]=size(shallow_image,1);
    % binary_shallow_image_adp=shallow_image;
    % binary_choroid_image_adp=choroid_image;
    % for i_slice=1:num_slice
    %     current_shallow=binary_shallow_image_adp(i_slice,:,:);
    %     current_choroid=binary_choroid_image_adp(i_slice,:,:);
    %     current_shallow=adaptthresh(current_shallow);
    %     current_choroid=adaptthresh(current_choroid);
    %     binary_shallow_image_adp(i_slice,:,:)=current_shallow;
    %     binary_choroid_image_adp(i_slice,:,:)=current_choroid;
    % end

    % fuse two images
    binary_image=binary_shallow_image+binary_choroid_image;
    binary_image(binary_image>=1)=1;

    % store outcome into struct
    p2.first_binary_image=binary_image;

    disp("First Image Binarization COMPLETED.")
end