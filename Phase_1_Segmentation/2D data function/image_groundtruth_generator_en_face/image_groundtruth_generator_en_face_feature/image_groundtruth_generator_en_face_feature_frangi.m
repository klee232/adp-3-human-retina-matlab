% Created by Kuan-Min Lee
% Createed date: May 23rd, 2024

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% Input parameter:
% data_storage: arrays of a series of 2D OCTA image

% Output:
% frangi_data_storage: frangi images arrays

function [frangi_data_storage]=image_groundtruth_generator_en_face_feature_frangi(data_storage)

    %% conduct frangi filter to extract vessel from each image
    
    num_files=size(data_storage,1);
    frangi_data_storage=zeros(size(data_storage));

    for i_file=1:num_files
        current_data_img=squeeze(data_storage(i_file,:,:));
        frangi_current_data_img=FrangiFilter2D(current_data_img);
        frangi_data_storage(i_file,:,:)=frangi_current_data_img;
    end

end