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


function [octa_gt_storage,oct_gt_storage, picture_obj_octa, picture_obj_oct]=image_groundtruth_refiner_rev(picture_obj_octa,picture_obj_oct)
    %% load the original image and groundtruth file for OCTA and OCT
    % check if the all the intended file existed
    if (isfile("temp_data/octa_data_gt_series.mat")) && ...
       (isfile("temp_data/oct_data_gt_series.mat")) && ...
       (isfile("temp_data/OCTA_data.mat")) && ...
       (isfile("temp_data/OCT_data.mat"))

       octa_gt_storage=picture_obj_octa.OCTA_img_gt;
       oct_gt_storage=picture_obj_oct.OCT_img_gt;
       octa_storage=picture_obj_octa.OCTA_img_den;
       oct_storage=picture_obj_oct.OCT_img_den;

    % if one of the intended file doesn't exist, return an error message
    else
        error("Error in file storage. Please double check the image groundtruth generator function");
    end

    %% initiate the indicator, start index, and end index for data loading
    % load the indicator file
    if (isfile("temp_data/indicator.mat"))
        load("temp_data/indicator.mat"); % variable name: indicator
    else
        indicator="";
    end
    % load the start index file 
    if (isfile("temp_data/index.mat"))
        load("temp_data/index.mat"); % variable name: i_slice
        start_ind=i_slice;
    else
        start_ind=1;
    end  
    % initiate end index
    end_ind=size(octa_gt_storage,1);
            

    %% looping through each slice in z direction
    for i_slice=start_ind:end_ind
        % loop through every slice
        disp(string(i_slice))
        current_series=squeeze(octa_gt_storage(i_slice,:,:)); % octa groundtruth series (used only in octa refiner)
        current_series_oct=squeeze(oct_gt_storage(i_slice,:,:)); % oct groundtruth series (used only in oct refiner)
        current_series_org_octa=squeeze(octa_storage(i_slice,:,:));
        current_series_org_oct=squeeze(oct_storage(i_slice,:,:)); % original oct series

        % different indictaor situation
        if indicator=="octa" || indicator==""
            % OCTA part
            % conduct binarization
            % this file will save
            % 1. slice index of 3D OCT
            % 2. medium terminate indicator (octa)
            [current_series,sign]=image_groundtruth_refiner_octa_rev(i_slice,current_series,current_series_org_octa,octa_gt_storage,octa_storage);
            octa_gt_storage(i_slice,:,:)=current_series;
            if sign=="break"
                save("temp_data/octa_data_gt_series.mat","octa_gt_storage");
                save("temp_data/oct_data_gt_series.mat","oct_gt_storage");
                save("temp_data/index.mat","i_slice");
                break;
            end

            % OCT part
            % conduct positive binarization (0 to 1)
            [current_series_oct,sign]=image_groundtruth_refiner_oct_rev(i_slice,current_series_oct,current_series_org_oct,oct_gt_storage,oct_storage);
            oct_gt_storage(i_slice,:,:)=current_series_oct;
            if sign=="break"
                save("temp_data/octa_data_gt_series.mat","octa_gt_storage");
                save("temp_data/oct_data_gt_series.mat","oct_gt_storage");
                save("temp_data/index.mat","i_slice");
                break;
            end

        elseif indicator=="oct" 
            % OCT part
            [current_series_oct,sign]=image_groundtruth_refiner_oct_rev(i_slice,current_series_oct,oct_gt_storage,oct_storage);
            oct_gt_storage(i_slice,:,:)=current_series_oct;
            if sign=="break"
                save("temp_data/octa_data_gt_series.mat","octa_gt_storage");
                save("temp_data/oct_data_gt_series.mat","oct_gt_storage");
                save("temp_data/index.mat","i_slice");
                break;
            end
        else
            error("wrong indicator detected. please double check")
        end

        % if this is the last slice, delete the current temp_data directory
        if i_slice==end_ind
            rmdir("temp_data","s");
        end
        
    end % ending for i_slice_z  

    % store outcome into struct
    picture_obj_octa.refined_octa=octa_gt_storage;
    picture_obj_oct.refined_oct=oct_gt_storage;
end
            

