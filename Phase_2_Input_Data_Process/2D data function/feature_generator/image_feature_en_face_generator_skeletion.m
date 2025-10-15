% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is built to train the implemented neural network

% Input Parameter:
% data_storage: segmentation data storage from previous process

% Output Parameter:
% skeleton_data_storage: skeleton data storage


function [skeleton_data_storage]=image_feature_en_face_generator_skeletion(data_storage)

    %% conduct skeletionizaton on each en face octa image
    num_files=size(data_storage,4);
    num_layers=size(data_storage,1);

    % create storage variable
    skeleton_data_storage=zeros(size(data_storage));

    % loop through each file
    for i_file=1:num_files
        current_data_img=squeeze(data_storage(:,:,:,i_file));

        % loop through each layer
        for i_layer=1:num_layers
            current_layer_data_img=logical(squeeze(current_data_img(i_layer,:,:)));
            skeleton_current_layer_data_img=bwskel(current_layer_data_img);
            skeleton_data_storage(i_layer,:,:,i_file)=skeleton_current_layer_data_img;
        end

    end


end
