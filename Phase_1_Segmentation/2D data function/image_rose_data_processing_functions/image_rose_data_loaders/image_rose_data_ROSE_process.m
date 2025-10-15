% Created by Kuan-Min Lee
% Created date: Jan. 9th, 2024 (last update: Jan. 10th 2024)
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This script file is created to process data augmentation and partition on ROSE dataset

% Input parameter:
% data_pathway: the directory that stores the data (String)
% filetype: the file type (for this case, image type) of the dataset
% (String)
% ratio: the ratio for validation dataset (float number<1)
% Output variable:
% train_path: the pathway storing the train dataset (String)
% valid_path: the pathway storing the validation dataset (String)
% test_path: the pathway storing the test dataset (String)
% train_image_array: the numeric array for training image (multi-dimensional array)
% valid_image_array: the numeric array for validation image (multi-dimensional array)
% test_image_array: the numeric array for testing image (multi-dimensional array)


function [train_image_array,valid_image_array]=image_rose_data_ROSE_process(data_pathway,filetype,ratio)
     
    %% Pre-check (checked and updated)
    % check if the input is for the right type
    path_check=isstring(data_pathway);
    if ~path_check
        error("MyComponent:incorrectType",...
                "Data pathway must be a string, not a %s.", class(data_pathway))
    end
    % check if the pathway exists
    path_exist_check=exist(data_pathway,'dir');
    if ~path_exist_check
        error("Data pathway does not exist. Please double check")
    end
    % check if the pathway is empty
    allfileType=strcat("*.",filetype);
    image_content=fullfile(data_pathway,allfileType); % in our case, the images that has been input are all .tif files
    tif_files=dir(image_content);
    num_files=length(tif_files);
    if num_files==0
        error("Data pathway is empty. Please check.")
    end


    %% Conduct Image Augmentation and Save It Back 
    % Input the number of data augmentation

    % Create variable for storing
    example_file=tif_files(1).name;
    example_file=fullfile(data_pathway,example_file);
    example_img=imread(example_file);
    [row_img,col_img,chn_img]=size(example_img);
    if chn_img>1
        convert_ind=true;
        example_img=im2gray(example_img);
        [row_img,col_img,~]=size(example_img);
    else
        convert_ind=false;
    end
    img_store=zeros(row_img,col_img,num_files);

    % Loop through each image
    % show up loading message
    for i_file=1:num_files
        current_file=tif_files(i_file).name;
        current_file_path=fullfile(data_pathway,current_file);
        fprintf(1,"Now reading %s\n", current_file_path);
        current_image=imread(current_file_path);
        current_image=double(current_image);
        if convert_ind
            current_image=im2gray(current_image);
        end

        img_store(:,:,i_file)=current_image;

    end


    %% partition the img_store into train and validation array
    total_file_num=num_files;
    valid_file_num=floor(ratio*total_file_num);
    train_file_num=total_file_num-valid_file_num;
    train_image_array=img_store(:,:,1:train_file_num);
    valid_image_array=img_store(:,:,(train_file_num+1):(train_file_num+valid_file_num));


end