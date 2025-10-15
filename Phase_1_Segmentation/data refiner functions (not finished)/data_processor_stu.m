function [octa_storage,octa_gt_storage,...
          oct_storage,oct_gt_storage,...
          all_data_obj_octa, all_data_obj_oct]=data_processor_stu()
    if ~exist("comp_data/","dir")
        clear all;
    end
    % include data directory
    addpath('dataset','end');

    path_file="~/data/klee232/s1/";
    file_name="filename_1.mat";
    full_filepath=strcat(path_file,file_name);

    % load all raw files inside dataset folders
    example_file=load(full_filepath);
    % OCT files
    file_type="*.mat";
    directory_file=strcat(path_file,file_type);
    all_oct_files=dir(directory_file);
    num_oct_files=length(all_oct_files);
    % create storage object for oct images
    oct_storage=zeros(num_oct_files,dep_size,row_size,col_size);
    for i_oct_file=1:num_oct_files
        current_oct_file=load(all_oct_files(i_oct_file).name);
        oct_storage(i_oct_file,:,:,:)=current_oct_file.II1;
    end
    % OCTA files
    all_octa_files=dir('dataset/*-OCTA.mat');
    num_octa_files=length(all_octa_files);
    % create storagge object for octa images
    octa_storage=zeros(num_octa_files,dep_size,row_size,col_size);
    for i_octa_file=1:num_octa_files
        current_octa_file=load(all_octa_files(i_octa_file).name);
        octa_storage(i_octa_file,:,:,:)=current_octa_file.DD1;
    end

    % create directory for processed data
    dataset_dir='processed_data';
    % if the current directory doesn't contain directory for processed
    % data, create one
    if ~isfolder(dataset_dir)
        mkdir (dataset_dir)
    end

    % launch data groundtruth generator and loop through each file
    addpath('Groundtruth generation/','end');
    % create storage object for groundtruth
    if ~exist("temp_data/","dir")
        oct_gt_storage=zeros(size(oct_storage));
        octa_gt_storage=zeros(size(octa_storage));
    else
        load("temp_data/octa_data_gt_series.mat");
        octa_gt_storage=zeros([num_octa_files size(filtered_octa_img_gt)]);
        octa_gt_storage(1,:,:,:)=filtered_octa_img_gt;
        load("temp_data/oct_data_gt_series.mat");
        oct_gt_storage=zeros([num_oct_files size(filtered_oct_img_gt)]);
        oct_gt_storage(1,:,:,:)=filtered_oct_img_gt;
        load("processed_data/picture_obj_octa.mat");
        load("processed_data/picture_obj_oct.mat");
    end
    all_data_obj_octa=struct;
    all_data_obj_oct=struct;

    for i_file=1:num_octa_files
        % read each file in the processed director
        current_octa_img=octa_storage(i_file,:,:,:);
        current_oct_img=oct_storage(i_file,:,:,:);
        current_octa_img=squeeze(current_octa_img);
        current_oct_img=squeeze(current_oct_img);
        % conduct segmentation octa and oct if no processed done yet
        if ~exist("temp_data/","dir")
            [current_octa_img_gt, current_oct_img_gt,current_octa_img_den,current_oct_img_den, picture_obj_octa, picture_obj_oct]=image_groundtruth_generator(current_octa_img,current_oct_img);            
        else
            current_octa_img_den=current_octa_img;
            % current_oct_img_den=current_oct_img;
            current_octa_img_gt=octa_gt_storage(i_file,:,:,:);
            current_octa_img_gt=squeeze(current_octa_img_gt);
            % current_oct_img_gt=oct_gt_storage(i_file,:,:,:);
            % current_oct_img_gt=squeeze(current_oct_img_gt);
        end

        % conduct segmentation refiner octa
        [current_octa_img_gt, picture_obj_octa]=image_groundtruth_refiner_octa(picture_obj_octa,current_octa_img_gt);
        % conduct segmentation refiner oct
        % [current_oct_img_gt, picture_obj_oct]=image_groundtruth_refiner_oct(picture_obj_oct,current_oct_img_gt);
        % store image groundtruth
        octa_gt_storage(i_file,:,:,:)=current_octa_img_gt;
        % oct_gt_storage(i_file,:,:,:)=current_oct_img_gt;
        % store denoised images
        octa_storage(i_file,:,:,:)=current_octa_img_den;
        % oct_storage(i_file,:,:,:)=current_oct_img_den;
        % store information
        all_data_obj_octa(i_file).info=picture_obj_octa;
        % all_data_obj_oct(i_file).info=picture_obj_oct;
    end

    % save the processed data inside the folder
    save("processed_data/octa_data.mat","octa_storage");
    save("processed_data/octa_gt_data.mat","octa_gt_storage");
    % save("processed_data/oct_data.mat","oct_storage");
    % save("processed_data/oct_gt_data.mat","oct_gt_storage");
end