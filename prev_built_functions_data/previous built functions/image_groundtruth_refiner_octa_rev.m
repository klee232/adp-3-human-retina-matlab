% Created by Kuan-Min Lee
% Createed date: Aug. 4th, 2024

% Brief User Introduction:
% This function is built to create User Interface for mannual selection for
% octa image

% Input parameter:
% current_octa_img_gt: original octa ground truth (3D image array)
% picture_obj: struct including image information

% Output:
% filtered_octa_img_gt: processed octa ground truth
% picture_obj: processed picture_obj


function [filtered_octa_img_gt,filtered_oct_img_gt, picture_obj_octa, picture_obj_oct]=image_groundtruth_refiner_octa_rev(picture_obj_octa,picture_obj_oct,current_octa_img_gt,current_oct_img)
    % create directory for storing intermediate file
    dataset_dir='temp_data';
    % if the current directory doesn't contain directory for processed
    % data, create one
    if ~isfolder(dataset_dir)
        mkdir (dataset_dir)
        % create outcome storage variable 
        filtered_octa_img_gt=current_octa_img_gt;
        % grab out only the region that contains content based on the previous
        % manually selected region
        z_ind=picture_obj_octa.z_ind;
        start_ind=z_ind(1);
        end_ind=z_ind(2);
    else
        if isfile("temp_data/octa_data_gt_series.mat") && isfile("temp_data/octa_data_gt_ind.mat")
            % if there exists a temp_data folder, load the file inside
            load("temp_data/octa_data_gt_series.mat");
            load("temp_data/octa_data_gt_ind.mat");
            load("temp_data/indicator.mat");
            % grab out only the region that contains content based on the previous
            % manually selected region
            z_ind=picture_obj_octa.z_ind;
            start_ind=i_slice;
            end_ind=z_ind(2);
        else
            filtered_octa_img_gt=current_octa_img_gt;
            z_ind=picture_obj.z_ind;
            start_ind=z_ind(1);
            end_ind=z_ind(2);
            indicator="";
        end
    end

    for i_slice=start_ind:end_ind
        % loop through every slice
        disp(string(i_slice))
        current_series=squeeze(filtered_octa_img_gt(i_slice,:,:));
        current_series_oct=squeeze(filtered_oct_img_gt(i_slice,:,:));
        current_series_org_oct=squeeze(current_oct_img(i_slice,:,:));
        % different indictaor situation
        if indicator=="octa-pos" || indicator==""
            % OCTA part
            % conduct positive binarization (0 to 1)
            [current_series]=image_groundtruth_refiner_octa_pos(i_slice,filtered_octa_img_gt,current_series,current_series_org_oct);
            filtered_octa_img_gt(i_slice,:,:)=current_series;
            % conduct negative binarization (1 to 0)
            [current_series]=image_groundtruth_refiner_octa_neg(i_slice,filtered_octa_img_gt,current_series,current_series_org_oct);
            filtered_octa_img_gt(i_slice,:,:)=current_series;
            % OCT part
            % conduct positive binarization (0 to 1)
            [current_series_oct]=image_groundtruth_refiner_oct_pos(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
            % conduct negative binarization (1 to 0)
            [current_series_oct]=image_groundtruth_refiner_octa_neg(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
        elseif indicator=="octa-neg" 
            % conduct negative binarization (1 to 0)
            [current_series]=image_groundtruth_refiner_octa_neg(i_slice,filtered_octa_img_gt,current_series,current_series_org_oct);
            filtered_octa_img_gt(i_slice,:,:)=current_series;
            % OCT part
            % conduct positive binarization (0 to 1)
            [current_series_oct]=image_groundtruth_refiner_oct_pos(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
            % conduct negative binarization (1 to 0)
            [current_series_oct]=image_groundtruth_refiner_octa_neg(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
        elseif indicator=="oct-pos" 
            % OCT part
            % conduct positive binarization (0 to 1)
            [current_series_oct]=image_groundtruth_refiner_oct_pos(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
            % conduct negative binarization (1 to 0)
            [current_series_oct]=image_groundtruth_refiner_octa_neg(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
        elseif indicator=="oct-neg" 
            % conduct negative binarization (1 to 0)
            [current_series_oct]=image_groundtruth_refiner_octa_neg(i_slice,filtered_oct_img_gt,current_series_oct,current_series_org_oct);
            filtered_oct_img_gt(i_slice,:,:)=current_series_oct;
        end
    end % ending for i_slice_z  

    % store outcome into struct
    picture_obj_octa.refined_octa=filtered_octa_img_gt;
    picture_obj_oct.refined_oct=filtered_oct_img_gt;
end
            

