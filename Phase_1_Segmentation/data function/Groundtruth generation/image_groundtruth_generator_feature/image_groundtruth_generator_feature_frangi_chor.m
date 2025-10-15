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


function  [picture_obj_octa,OCTA_img_frangi_choroid]=image_groundtruth_generator_feature_frangi_chor(picture_obj_octa,enhanced_filtered_OCTA_img_chor,vessel_parameters)   

   %% Conduct Frangi filtering
   t_tic = tic;
   frangi_OCTA_choroid=frangi_filter(enhanced_filtered_OCTA_img_chor, vessel_parameters);
   fprintf("This implementation on CPU takes %.2f seconds\n", toc(t_tic))   
    

    %% store the enhanced outcome
    OCTA_img_frangi_choroid=frangi_OCTA_choroid;
    picture_obj_octa.filtered_img_rdhe_frangi_choroid=OCTA_img_frangi_choroid;

end