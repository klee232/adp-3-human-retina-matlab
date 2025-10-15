% Created by Kuan-Min Lee
% Created date: Dec. 8th, 2023
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to angle rotations (data
% augmentation) for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% num_aug: number of data augmentation (integer)
% image: training image that is intended to create data augmentation
% (multi-dimentional array)
% img_index: current image index 
% (used to check if this is a training or validation image) (integer)
% valid_index: the start index of validation image 
% (used to check if this is a training or validation image) (integer)
% filetype: filetype of the current image (string)
% root_dir: the directory of current image (string)
% Output Parameter
% out_dir: the directory of the output image (string)

function [out_img,out_gt]=data_augmentor(num_aug,image,gt,img_index,valid_index,filetype,gt_filetype,root_dir)
    % Pre-check
    % check if it's at least a 2D array type variable
    [row,col,~]=size(image);
    if (row<2) || (col<2)
        error("The input variable is not a 2D array. Please input another variable.")
    end
    % Conduct image rotation on each image and save it back
    num_ang=num_aug; % in this case, the augmentation is set to enlarge the original dataset by four times
    angles=randi([-10 10],1,num_aug);
    % create an array for storing the output image array
    [row_img,col_img,~]=size(image);
    out_img=zeros(row_img,col_img,num_ang,class(image));
    out_gt=zeros(row_img,col_img,num_ang,class(gt));
    for i_ang=1:num_ang
        % Conduct image rotation
        current_img_rot=imrotate(image,angles(i_ang),"nearest","crop");
        current_gt_rot=imrotate(gt,angles(i_ang),"nearest","crop");
        % store the current image into the output array
        out_img(:,:,i_ang)=current_img_rot;
        out_gt(:,:,i_ang)=current_gt_rot;
        ang_ind=num2str(i_ang);
        % store assigned image and groundtruth
        % if it's a training data
        if contains(root_dir,"\train")
            % if it's a valid data
            if img_index>=valid_index
                if ~exist("valid","dir")
                    img_path=strcat(fileparts(root_dir),"\valid\img");
                    gt_path=strcat(fileparts(root_dir),"\valid\gt");
                    mkdir (img_path)
                    mkdir (gt_path)
                end
                save_path_valid=strcat(fileparts(root_dir),"\valid\img\");
                save_path_valid_gt=strcat(fileparts(root_dir),"\valid\gt\");
                save_filename=strcat(save_path_valid,string(img_index),"_raw_", string(ang_ind),"_",string(img_index),filetype);
                imwrite(current_img_rot,save_filename);
                save_filename=strcat(save_path_valid_gt,string(img_index),"_gt_", string(ang_ind),"_",string(img_index),gt_filetype);
                imwrite(current_gt_rot,save_filename);
            else
                save_path=strcat(root_dir,"\img\");
                save_path_gt=strcat(root_dir,"\gt\");
                save_filename=strcat(save_path,string(img_index),"_raw_", string(ang_ind),"_",string(img_index),filetype);
                imwrite(current_img_rot,save_filename);
                save_filename=strcat(save_path_gt,string(img_index),"_gt_", string(ang_ind),"_",string(img_index),gt_filetype);
                imwrite(current_gt_rot,save_filename);
            end
        % if it's a testing data
        elseif contains(root_dir,"\test")
            save_path=strcat(root_dir,"\img\");
            save_path_gt=strcat(root_dir,"\gt\");
            save_filename=strcat(save_path,string(img_index),"_raw_", string(ang_ind),"_",string(img_index),filetype);
            imwrite(current_img_rot,save_filename);
            save_filename=strcat(save_path_gt,string(img_index),"_gt_", string(ang_ind),"_",string(img_index),gt_filetype);
            imwrite(current_gt_rot,save_filename);
        else
            error("Unknown datatype (training, valid, or testing. Please double check")
        end


    end
       
end