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
% choroid_image: OCTA image with only choroid layer (3D image array)
% choroid_thres_ratio: threshold for binarization 

% Output:
% p2: struct to store the process information
% binary_choroid_image: outcome of binarization image for choroid layer (binarized 3D image) 

% optimal setting:
% choroid_thres_ratio=0.85; % filtered out 5% smallest pixels (tuned)


function [p2,binary_choroid_image]=image_groundtruth_generator_binary_chor(p2,choroid_image,choroid_thres_ratio)
    disp("First Image Binarization IS RUNNING ...")
  
    % choroid leayer
    flat_choroid_image=choroid_image(:);
    flat_choroid_image=flat_choroid_image(flat_choroid_image>0);
    sorted_choroid_array=sort(flat_choroid_image,'descend');
    choroid_thres_ind=floor((1-choroid_thres_ratio)*length(flat_choroid_image(:)));
    if choroid_thres_ind==0
        binary_choroid_image=0;
    else
        choroid_thres=sorted_choroid_array(choroid_thres_ind);
        binary_choroid_image=choroid_image;
        binary_choroid_image(choroid_image>choroid_thres)=1;
        binary_choroid_image(choroid_image<=choroid_thres)=0;
    end
    

    %% store outcome into struct
    p2.binary_img_rdhe_frangi_choroid=binary_choroid_image;
    disp("First Image Binarization COMPLETED.")
end