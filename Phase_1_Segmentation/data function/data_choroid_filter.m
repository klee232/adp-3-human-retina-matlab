% Created by Kuan-Min Lee
% Createed date: Oct. 22nd, 2024

% Brief User Introduction:
% This function is built to exclude out the choroid layer from the original octa and octa groundtruth image 

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]
% 4D image array format: [num_file, num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% octa_storage: processed image storage for OCTA image (4D image array)
% octa_gt_storage: processed image storage for OCTA groundtruth image (4D image array)

% Output:
% octa_gt_storage: padded processed octa groundtruth storage for all files (4D image
% array)
% octa_storage: padded processed octa image storage for all files (4D image
% array)



function [filtered_octa_storage,filtered_octa_gt_storage]=data_choroid_filter(octa_storage,octa_gt_storage)
        
    %% find out all objective files
    % find out how many files in picture_obj folder
    folder_path="/oscar/home/klee232/data/klee232/processed_data/picture_obj/";
    oct_obj_files=dir(fullfile(folder_path,'*-OCT_obj.mat'));
    
    %% loop through each file and filter the images based on the extracted mask
    num_file=size(octa_storage,1);
    filtered_octa_storage=cell(num_file,1);
    filtered_octa_gt_storage=cell(num_file,1);
    for i_file=1:num_file
        % grab out the current mask
        current_oct_obj_file_path=strcat(folder_path,oct_obj_files(i_file).name);
        current_oct_obj_file=load(current_oct_obj_file_path);
        current_oct_seg_img=current_oct_obj_file.picture_obj_oct.filtered_mask_seg;

        % make all value 14 as zeros
        mask_choroid=ones(size(current_oct_seg_img));
        mask_choroid(current_oct_seg_img==14)=0;

        % filtered out the image using the created mask
        current_octa_img=octa_storage{i_file,1};
        current_octa_gt_img=octa_gt_storage{i_file,1};
        current_octa_img=cell2mat(current_octa_img);
        current_octa_gt_img=cell2mat(current_octa_gt_img);
        filtered_current_octa_img=current_octa_img.*mask_choroid;
        filtered_current_octa_gt_img=current_octa_gt_img.*mask_choroid;

        % store the resulted image
        filtered_octa_storage{i_file,1}=filtered_current_octa_img;
        filtered_octa_gt_storage{i_file,1}=filtered_current_octa_gt_img;
        
    end



end