% Created by Kuan-Min Lee
% Createed date: May 23rd, 2024

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% Input parameter:

% Output:
% octa_storage: storage for original OCTA image
% octa_gt_storage: groundturth for OCTA image


function [en_face_octa_seg_storage]=image_groundtruth_generator_en_face(octa_surf_storage,octa_deep_storage,octa_choroid_storage)

    %% conduct image denoising
    % denoised surface capillary layer (checked)
    region_size=[5,5];
    spatial_var=0.01;
    spatial_sigma=7;
    [denoised_octa_surf_storage]=image_groundtruth_generator_denoise_filter_en_face(octa_surf_storage,region_size,spatial_var,spatial_sigma);

    % denoised deep capillary layer (checked)
    region_size=[5,5];
    spatial_var=0.004;
    spatial_sigma=5;
    [denoised_octa_deep_storage]=image_groundtruth_generator_denoise_filter_en_face(octa_deep_storage,region_size,spatial_var,spatial_sigma);

    % denoised choroid capillary layer
    region_size=[9,9];
    [denoised_octa_choroid_storage]=image_groundtruth_generator_denoise_filter_en_face(octa_choroid_storage,region_size,spatial_var,spatial_sigma); 

    %% conduct image edge detection
    % conduct canny edge detection
    % surface capillary layer
    [edge_octa_surf_storage]=image_groundtruth_generator_en_face_edge_canny(denoised_octa_surf_storage);

    % deep capillary layer
    [edge_octa_deep_storage]=image_groundtruth_generator_en_face_edge_canny(denoised_octa_deep_storage);

    % choroid capillary layer
    [edge_octa_choroid_storage]=image_groundtruth_generator_en_face_edge_canny(denoised_octa_choroid_storage);


    %% conduct image feature extraction
    % conduct frangi feature extraction
    % surface capillary layer
    [frangi_octa_surf_storage]=image_groundtruth_generator_en_face_feature_frangi(denoised_octa_surf_storage);

    % deep capillary layer
    [frangi_octa_deep_storage]=image_groundtruth_generator_en_face_feature_frangi(denoised_octa_deep_storage);

    % choroid capillary layer
    [frangi_octa_choroid_storage]=image_groundtruth_generator_en_face_feature_frangi(denoised_octa_choroid_storage);


    %% conduct image fusion (add edge with original denoising image, and multiply with inverse frangi feature)
    % surface capillary layer
    [fuse_octa_surf_storage]=image_groundtruth_generator_en_face_fuse(denoised_octa_surf_storage, edge_octa_surf_storage, frangi_octa_surf_storage);
    
    % deep capillary layer
    [fuse_octa_deep_storage]=image_groundtruth_generator_en_face_fuse(denoised_octa_deep_storage, edge_octa_deep_storage, frangi_octa_deep_storage);
    
    % choroid capillary layer
    [fuse_octa_choroid_storage]=image_groundtruth_generator_en_face_fuse(denoised_octa_choroid_storage, edge_octa_choroid_storage, frangi_octa_choroid_storage);
    

    %% conduct image binarization
    % conduct image binarization 
    % surface capillary layer
    [binary_octa_surf_storage]=image_groundtruth_generator_en_face_binary_global(fuse_octa_surf_storage);

    % deep capillary layer
    [binary_octa_deep_storage]=image_groundtruth_generator_en_face_binary_global(fuse_octa_deep_storage);

    % choroid capillary layer
    [binary_octa_choroid_storage]=image_groundtruth_generator_en_face_binary_global(fuse_octa_choroid_storage);


    %% conduct image binarization closing
    % surface capillary layer
    [close_octa_surf_data_storage]=image_groundtruth_generator_en_face_close(binary_octa_surf_storage);

    % deep capillary layer
    [close_octa_deep_data_storage]=image_groundtruth_generator_en_face_close(binary_octa_deep_storage);

    % choroid capillary layer
    [close_octa_choroid_data_storage]=image_groundtruth_generator_en_face_close(binary_octa_choroid_storage);


    %% store final outcome
    num_files=size(octa_surf_storage,1);

    en_face_octa_seg_storage=zeros(3, size(octa_surf_storage,2),size(octa_surf_storage,3),num_files);
    for i_file=1:num_files
        current_close_octa_surf_img=close_octa_surf_data_storage(i_file,:,:,:);
        current_close_octa_deep_img=close_octa_deep_data_storage(i_file,:,:);
        current_close_octa_choroid_img=close_octa_choroid_data_storage(i_file,:,:);

        en_face_octa_seg_storage(1,:,:,i_file)=current_close_octa_surf_img;
        en_face_octa_seg_storage(2,:,:,i_file)=current_close_octa_deep_img;
        en_face_octa_seg_storage(3,:,:,i_file)=current_close_octa_choroid_img;
    end





end