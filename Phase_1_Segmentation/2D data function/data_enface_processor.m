% Created by Kuan-Min Lee
% Createed date: Feb. , 2025

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 2D en face OCTA image network. The below function executes the image
% groundtruth generator en face function under the
% image_groundtruth_generator_en_face folder.

% Input parameter:

% Output:
% en_face_octa_seg_storage: en face segmentation (one file contains all
% surface, deep, and choroid layer)

function [en_face_octa_seg_storage]=data_enface_processor()
    
    disp("octa data extration running...");

    %% load file if the en face images have been created
    % if not exist, send out error message and end the function
    if ~exist("~/data/klee232/processed_data/octa en face arrays/octa_data_surface_en_face.mat") || ...
       ~exist("~/data/klee232/processed_data/octa en face arrays/octa_data_deep_en_face.mat") || ...
       ~exist("~/data/klee232/processed_data/octa en face arrays/octa_data_choroid_en_face.mat")
        error("En face file is not completed. Please double check");
    % if exist load the file
    else
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_surface_en_face.mat"); 
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_deep_en_face.mat"); 
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_choroid_en_face.mat"); 
    end


    %% conduct groundtruth generator
    [en_face_octa_seg_storage]=image_groundtruth_generator_en_face(octa_surf_storage,octa_deep_storage,octa_choroid_storage);


    %% save the processed file
    save("~/data/klee232/processed_data/octa arrays/octa_gt_data_en_face.mat","en_face_octa_seg_storage",'-v7.3');


end