% Created by Kuan-Min Lee
% Created date:: Jan. 6th, 2024

% Brief User Introduction:
% (revised from Morgan collab)
% This module utilize a
% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% picture_obj_octa: struct to store the process information
% filtered_OCTA_image: input 3D image (3D image array)
% filtered_seg_mask: mask for segmentation layer (3D mask array)

% Output:
% picture_obj_octa: struct to store the process information (struct)
% OCTA_img_den: outcome of denoised image (denoised 3D image) 


function  [picture_obj_octa,OCTA_img_frangi]=image_groundtruth_generator_feature_frangi_combined(picture_obj_octa,OCTA_img,vessel_parameters)
    
    %% Conduct Frangi filtering
    % surface layer
    t_tic = tic;
    frangi_OCTA = frangi_filter(OCTA_img, vessel_parameters);
    fprintf("This implementation on CPU takes %.2f seconds\n", toc(t_tic))


    %% store the outcome
    OCTA_img_frangi=frangi_OCTA;
    picture_obj_octa.filtered_img_rdhe_frangi=OCTA_img_frangi;

end