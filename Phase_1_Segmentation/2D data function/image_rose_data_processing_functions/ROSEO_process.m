% Created by Kuan-Min Lee
% Created date: Jan. 12th, 2024 (unfinisheed)
% All rights reserved to Leelab.ai

% Brief User Introduction:

% Input parameter:


function [train_path,valid_path,test_path]=ROSEO_process(data_pathway,filetype,ratio)
     
    % Pre-check
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

    % Set Default Value for train_path, valid_path, and test_path
    train_path="none";
    valid_path="none";
    test_path="none";


    % Conduct Image Augmentation and Save It Back 
    save_filetype=strcat(".",filetype);
    % Input the number of data augmentation
    num_angs=3;
    % Get the number of validation data number
    valid_ind=floor(num_files*ratio);
    % Create variable for storing
    example_file=tif_files(1).name;
    example_file=fullfile(data_pathway,example_file);
    example_img=imread(example_file);
    [row_img,col_img,chn_img]=size(example_img);
    img_store=zeros(num_angs*num_files,row_img,col_img,chn_img);
    % Loop through each image
    for i_file=1:num_files
        current_file=tif_files(i_file).name;
        current_file_path=fullfile(data_pathway,current_file);
         % show up loading message
        fprintf(1,"Now reading %s\n", current_file_path);
        current_image=imread(current_file_path);
        % conduct image augmentation (rotation in this case)
        [out_dir,out_img]=new_img_rotator(num_angs,current_image,i_file,valid_ind,save_filetype,data_pathway);
        % store the output image array
        img_store_ind_str=num_angs*(i_file-1)+1;
        img_store_ind_end=num_angs*(i_file-1)+num_angs;
        img_store(img_store_ind_str:img_store_ind_end,:,:,:)=out_img;
        % if it's a training data
        if contains(data_pathway,"train\")
            % if it does not hit validation threshold, then assign it to
            % training path
            if i_file<valid_ind
                train_path=out_dir;
            % if it does, then assign it to validation path
            else
                valid_path=out_dir;
            end
        % if it's a testing data, then assign it to testing path
        else
            test_path=out_dir;
        end
    end

    % store image store as .mat file
    % if this dataset is a training dataset
    if contains(data_pathway,"train\")
        % partition the dataset into training and validation dataset
        train_img_store=img_store(1:num_angs*(valid_ind-2)+num_angs,:,:,:);
        valid_img_store=img_store(num_angs*(valid_ind-1)+1:end,:,:,:);
        % store image store to assigned folder
        train_img_filename=strcat(train_path);

    % if it's a testing data, no need to partition it
    else
        test_path=out_dir;
    end


end