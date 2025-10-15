% Created by Kuan-Min Lee
% Createed date: Feb. 8th, 2025

% Brief User Introduction:
% This function is built to concatenate the surface capillary, deep
% capillary, and choroid layer together.

% Input parameter:
% octa_surf_storage: surface capillary storage
% octa_deep_storage: deep capillary storage
% octa_choroid_storage: octa_choroid_storage

% Output:
% en_face_octa_seg_storage: en face segmentation (one file contains all
% surface, deep, and choroid layer)


function [en_face_octa_storage]=data_en_face_concatenator(octa_surf_storage,octa_deep_storage,octa_choroid_storage)
    
    %% permute the dimension of the storage to fit the output dimension
    octa_surf_storage=permute(octa_surf_storage,[2,3,1]);
    octa_deep_storage=permute(octa_deep_storage,[2,3,1]);
    octa_choroid_storage=permute(octa_choroid_storage,[2,3,1]);


    %% output the concatenated output and save it
    num_files=size(octa_surf_storage,3);
    en_face_octa_storage=zeros(3,size(octa_surf_storage,1),size(octa_surf_storage,2),num_files);
    en_face_octa_storage(1,:,:,:)=octa_surf_storage;
    en_face_octa_storage(2,:,:,:)=octa_deep_storage;
    en_face_octa_storage(3,:,:,:)=octa_choroid_storage;

    save("~/data/klee232/processed_data/octa en face arrays/octa_data_en_face.mat","en_face_octa_storage",'-v7.3');

end