% Created by Kuan-Min Lee
% Created date: Jan. 31st, 2025
% All rights reserved to Leelab.ai

% Brief User Introducttion:

% Input Parameter


% Output Parameter


function [rotated_data_storage, rotated_label_storage]=image_augmentation_rotator(data_storage, label_storage)
    
    %% conduct 3D rotation for x, y, and z axis
    rotation_angle=0:10:30; % rotational angles (this angles will be used in x, y, and z axis)

    % create storage variable
    rotated_data_storage=zeros(size(data_storage,1),size(data_storage,2), size(data_storage,3), 3*size(rotation_angle,2)*size(data_storage,4), size(data_storage,5));
    rotated_label_storage=zeros(3*size(rotation_angle,2)*size(label_storage,1),size(label_storage,2));

    % conduct rotation on each file
    num_file=size(data_storage,4);
    num_rotation=size(rotation_angle,2);
    num_axis=3;
    num_fold=size(data_storage,5);
    for i_fold=1:num_fold
        for i_file=1:num_file
            % grab out current file
            current_data_img=data_storage(:,:,:,i_file,i_fold);
            current_data_img=squeeze(current_data_img);
            current_label=label_storage(i_file,i_fold);
            rotation_axis=zeros(1,3);
            for i_axis=1:num_axis
                for i_rotation=1:num_rotation
                    % conduct image rotation
                    rotation_axis(1,i_axis)=1;
                    current_rotation_angle=rotation_angle(1,i_rotation);
                    current_rotated_data_img=imrotate3(current_data_img,current_rotation_angle,rotation_axis,"nearest","loose");

                    % store back resulted image
                    rotated_data_storage(:,:,:,(i_file-1)*(num_axis*num_rotation)+(i_axis-1)*num_rotation+i_rotation,i_fold)=current_rotated_data_img;
                    rotated_label_storage((i_file-1)*(num_axis*num_rotation)+(i_axis-1)*num_rotation+i_rotation,i_fold)=current_label;

                end

            end

        end
        
    end

end