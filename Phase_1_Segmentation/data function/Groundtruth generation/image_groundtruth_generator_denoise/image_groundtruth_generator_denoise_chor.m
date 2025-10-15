% Created by Kuan-Min Lee
% Createed date: Oct. 6th, 2024

% Brief User Introduction:
% The following function is utilized to denoise choroid layer 3d image (in
% this case, initialize the regional dynaimc histogram equalization

% input argument:
% picture_obj_octa: picture object for octa storage (struct)
% filtered_OCTA_img: 3D image array (3D array)
% filtered_orig_maks: layer mask array (3D array)

% output:
% picture_obj_octa: picture object for octa storage (struct)
% OCTA_img_den_chor: output 3D array with enhanced contrast (3D array)


function [picture_obj_octa,OCTA_img_den_chor]=image_groundtruth_generator_denoise_chor(picture_obj_octa,filtered_OCTA_img,filtered_seg_mask)
    %% grab out the choroid layer image
    mask_choroid_filtered=zeros(size(filtered_seg_mask));
    mask_choroid_filtered(filtered_seg_mask==14)=1;
    filtered_OCTA_img_chor=filtered_OCTA_img.*mask_choroid_filtered;

    %% initialize regional dynamic histogram equalization
    region_size=[1,16,16];
    enhanced_filtered_OCTA_img_chor=image_groundtruth_rdhe_3d(filtered_OCTA_img_chor,region_size);

    %% image closing
    % % conduct image closing
    % se = strel('disk',1); % tuned
    % OCTA_choroid_edge_horz=imclose(OCTA_choroid_edge_horz,se);
    % OCTA_choroid_edge_vert_1=imclose(OCTA_choroid_edge_vert_1,se);
    % OCTA_choroid_edge_vert_2=imclose(OCTA_choroid_edge_vert_2,se);
    % 
    % % fuse the edge detection outcomes
    % OCTA_choroid_edge=sqrt(OCTA_choroid_edge_horz.^2+sqrt(OCTA_choroid_edge_vert_1.^2+OCTA_choroid_edge_vert_2.^2).^2);
    % OCTA_choroid_edge(OCTA_choroid_edge<1.414)=0;

    %% store the enhanced outcome
    OCTA_img_den_chor=enhanced_filtered_OCTA_img_chor;
    picture_obj_octa.filtered_img_den_chor=OCTA_img_den_chor;

end