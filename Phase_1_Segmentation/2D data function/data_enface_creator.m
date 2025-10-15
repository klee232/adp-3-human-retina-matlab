% Created by Kuan-Min Lee
% Created date: Feb. 3rd, 2025
% All rights reserved to Leelab.ai


% Brief User Introduction:
% This function is built to create en face images for surface, deep capillary layer and choroid layer from 3D images


% Input Parameter:


% Output Parameter:
% octa_surf_storage: storage for 2D en face octa surface capillary layer
% octa_deep_storage: storage for 2D en face octa deep capillary layer
% octa_choroid_storage: storage for 2D en face octa choroid capillary layer


function [octa_surf_storage,octa_deep_storage,octa_choroid_storage]=data_enface_creator()
    
    %% find out the picture objects stored from previous 3D processes
    pict_obj_dir="~/data/klee232/processed_data/picture_obj/";
    pict_obj_type="*-OCTA_complete_obj.mat";
    all_files=dir(strcat(pict_obj_dir,pict_obj_type));


    %% grab out each of the picture object files and create en-face images based on corresponding mask
    % (2-6): surface capillary
    % (8): deep capillary
    % (14): choroid layer

    % find out the number of files
    num_files=numel(all_files);

    % create storage variable
    load(strcat(pict_obj_dir,all_files(1).name));
    example_file=picture_obj_octa.filtered_img;
    octa_surf_storage=zeros(num_files,size(example_file,2),size(example_file,3));
    octa_deep_storage=zeros(num_files,size(example_file,2),size(example_file,3));
    octa_choroid_storage=zeros(num_files,size(example_file,2),size(example_file,3));

    for i_file=1:num_files
        % load current file
        current_file=all_files(i_file).name;
        load(strcat(pict_obj_dir,current_file));

        % grab out the corresponding octa image and mask
        current_octa=picture_obj_octa.filtered_img;
        current_mask=picture_obj_octa.filtered_mask_seg;

        % create mask for filtering out everything except surface capillary
        % layer
        mask_surf=zeros(size(current_mask));
        mask_surf(current_mask>=2 & current_mask<=6)=1;

        % create mask for filtering out everything except deep capillary
        % layer
        mask_deep=zeros(size(current_mask));
        mask_deep(current_mask==8)=1;

        % create mask for filtering out everything except for choroid layer
        mask_choroid=zeros(size(current_mask));
        mask_choroid(current_mask==14)=1;
        
        % create surface, deep, and choroid layer 3d image
        current_octa_surf=current_octa.*mask_surf;
        current_octa_deep=current_octa.*mask_deep;
        current_octa_choroid=current_octa.*mask_choroid;

        % conduct 2D en face compression
        current_octa_surf_en_face=squeeze(max(current_octa_surf,[],1));
        current_octa_deep_en_face=squeeze(max(current_octa_deep,[],1));
        current_octa_choroid_en_face=squeeze(max(current_octa_choroid,[],1));
        
        % store back to storage variable
        octa_surf_storage(i_file,:,:)=current_octa_surf_en_face;
        octa_deep_storage(i_file,:,:)=current_octa_deep_en_face;
        octa_choroid_storage(i_file,:,:)=current_octa_choroid_en_face;

    end


    %% save the the processed enface images
    save("~/data/klee232/processed_data/octa arrays/octa_data_surface_en_face.mat","octa_surf_storage",'-v7.3');
    save("~/data/klee232/processed_data/octa arrays/octa_data_deep_en_face.mat","octa_deep_storage",'-v7.3');
    save("~/data/klee232/processed_data/octa arrays/octa_data_choroid_en_face.mat","octa_choroid_storage",'-v7.3');


end