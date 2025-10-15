% Created by Kuan-Min Lee
% Createed date: Jan. 22nd 2025

% Brief User Introduction:
% This function is built to extract the oct and oct segmentation image from
% the stored objective files

% Input parameter:

% Output:
% oct_storage: storage for original OCT image
% oct_seg_storage: groundturth for OCT segmentation image


function [oct_storage,oct_seg_storage]=data_processor_oct()
    
    disp("oct data extration running...");
    

    %% extract the filtered oct image and oct segmentation image
    % find out how many files in picture_obj folder
    folder_path="/oscar/home/klee232/data/klee232/processed_data/picture_obj/";
    oct_obj_files=dir(fullfile(folder_path,'*-OCT_obj.mat'));
    num_oct_obj_files=numel(oct_obj_files);

    % loop through each file and grab out the objective images
    oct_storage=cell(num_oct_obj_files,1);
    oct_seg_storage=cell(num_oct_obj_files,1);
    for i_oct_obj_file=1:num_oct_obj_files
        % grab out the current oct objective file
        current_oct_obj_file_path=strcat(folder_path,oct_obj_files(i_oct_obj_file).name);
        current_oct_obj_file=load(current_oct_obj_file_path);

        % grab out the current oct image and oct segmentation image
        current_oct_img=current_oct_obj_file.picture_obj_oct.filtered_img;
        current_oct_seg_img=current_oct_obj_file.picture_obj_oct.filtered_mask_seg;

        % store the current oct image and oct segmentation image into
        % storage variable
        oct_storage{i_oct_obj_file}=current_oct_img;
        oct_seg_storage{i_oct_obj_file}=current_oct_seg_img;

    end


    % save the processed data inside the folder
    save("~/data/klee232/processed_data/oct_data.mat","oct_storage",'-v7.3');
    save("~/data/klee232/processed_data/oct_seg_data.mat","oct_seg_storage","-v7.3");

end