% Created by Kuan-Min Lee
% Created date: Jan. 10th, 2024 (last updated: Jan. 15th, 2024)
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This script is created to initialize the data preprocessing of ROSE
% dataset

% Input Parameter: None
% Output Parameter: None

function rose_data_saving()
    % loading ROSE dataset
    Rose_path="C:\Users\klee232\Desktop\Thesis\Codes\My works\data\ROSE";

    % Rose-1
    Rose_1_path=strcat(Rose_path,"\","ROSE-1");
    % SVC-DVC
    Rose_1_SDVC_path=strcat(Rose_1_path,"\","SVC_DVC");
    % training
    img_filetype='*.png';
    gt_filetype='*.tif';
    Rose_1_SDVC_train_path=strcat(Rose_1_SDVC_path,"\","train");
    Rose_1_SDVC_train_img_path=strcat(Rose_1_SDVC_train_path,"\","img"); % raw image
    Rose_1_SDVC_train_gt_path=strcat(Rose_1_SDVC_train_path,"\","gt"); % groundtruth image
    SDVC_train=fullfile(Rose_1_SDVC_train_img_path,img_filetype);
    SDVC_train=dir(SDVC_train);
    SDVC_gt=fullfile(Rose_1_SDVC_train_gt_path,gt_filetype);
    SDVC_gt=dir(SDVC_gt);
    % valid
    Rose_1_SDVC_valid_path=strcat(Rose_1_SDVC_path,"\","valid");
    Rose_1_SDVC_valid_img_path=strcat(Rose_1_SDVC_valid_path,"\","img"); % raw image
    Rose_1_SDVC_valid_gt_path=strcat(Rose_1_SDVC_valid_path,"\","gt"); % groundtruth image
    SDVC_valid=fullfile(Rose_1_SDVC_valid_img_path,img_filetype);
    SDVC_valid=dir(SDVC_valid);
    SDVC_valid_gt=fullfile(Rose_1_SDVC_valid_gt_path,gt_filetype);
    SDVC_valid_gt=dir(SDVC_valid_gt);
    % testing
    Rose_1_SDVC_test_path=strcat(Rose_1_SDVC_path,"\","test");
    Rose_1_SDVC_test_img_path=strcat(Rose_1_SDVC_test_path,"\","img"); % raw image
    Rose_1_SDVC_test_gt_path=strcat(Rose_1_SDVC_test_path,"\","gt"); % groundtruth image
    SDVC_test=fullfile(Rose_1_SDVC_test_img_path,img_filetype);
    SDVC_test=dir(SDVC_test);
    SDVC_test_gt=fullfile(Rose_1_SDVC_test_gt_path,gt_filetype);
    SDVC_test_gt=dir(SDVC_test_gt);
    % store each image (training)
    example_path=fullfile(Rose_1_SDVC_train_img_path,SDVC_train(1).name);
    example_path_gt=fullfile(Rose_1_SDVC_train_gt_path,SDVC_gt(1).name);
    example_file=imread(example_path);
    example_file_gt=imread(example_path_gt);
    num_files=length(SDVC_train);
    [row,col,~]=size(example_file);
    type=class(example_file);
    type_gt=class(example_file_gt);
    train_image_storage=zeros(row,col,num_files,type);
    gt_image_storage=zeros(row,col,num_files,type_gt);
    % store each image (training)
    for i_file=1:length(SDVC_train)
        current_train=SDVC_train(i_file).name;
        current_gt=SDVC_gt(i_file).name;
        full_current_train=fullfile(Rose_1_SDVC_train_img_path,current_train);
        full_current_gt=fullfile(Rose_1_SDVC_train_gt_path,current_gt);
        current_train_image=imread(full_current_train);
        current_train_image=rgb2gray(current_train_image);
        current_train_gt=imread(full_current_gt);
        train_image_storage(:,:,i_file)=current_train_image;
        gt_image_storage(:,:,i_file)=current_train_gt;
    end
    % store each image (valid)
    num_valid_files=length(SDVC_valid);
    valid_image_storage=zeros(row,col,num_files,type);
    valid_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_valid_files
        current_valid=SDVC_valid(i_file).name;
        current_valid_gt=SDVC_valid_gt(i_file).name;
        full_current_valid=fullfile(Rose_1_SDVC_valid_img_path,current_valid);
        full_current_valid_gt=fullfile(Rose_1_SDVC_valid_gt_path,current_valid_gt);
        current_valid_image=imread(full_current_valid);
        current_valid_image=rgb2gray(current_valid_image);
        current_valid_gt=imread(full_current_valid_gt);
        valid_image_storage(:,:,i_file)=current_valid_image;
        valid_gt_image_storage(:,:,i_file)=current_valid_gt;
    end
    % store each image (test)
    num_test_files=length(SDVC_test);
    test_image_storage=zeros(row,col,num_files,type);
    test_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_test_files
        current_test=SDVC_test(i_file).name;
        current_test_gt=SDVC_test_gt(i_file).name;
        full_current_test=fullfile(Rose_1_SDVC_test_img_path,current_test);
        full_current_test_gt=fullfile(Rose_1_SDVC_test_gt_path,current_test_gt);
        current_test_image=imread(full_current_test);
        current_test_image=rgb2gray(current_test_image);
        current_test_gt=imread(full_current_test_gt);
        test_image_storage(:,:,i_file)=current_test_image;
        test_gt_image_storage(:,:,i_file)=current_test_gt;
    end
    % save each file
    save("processed_mat\SDVC_train_img.mat","train_image_storage")
    save("processed_mat\SDVC_train_gt.mat","gt_image_storage")
    save("processed_mat\SDVC_valid_img.mat","valid_image_storage")
    save("processed_mat\SDVC_valid_gt.mat","valid_gt_image_storage")
    save("processed_mat\SDVC_test_img.mat","test_image_storage")
    save("processed_mat\SDVC_test_gt.mat","test_gt_image_storage")
    % DVC
    Rose_1_DVC_path=strcat(Rose_1_path,"\","DVC");
    % training
    img_filetype='*.tif';
    gt_filetype='*.tif';
    Rose_1_DVC_train_path=strcat(Rose_1_DVC_path,"\","train");
    Rose_1_DVC_train_img_path=strcat(Rose_1_DVC_train_path,"\","img"); % raw image
    Rose_1_DVC_train_gt_path=strcat(Rose_1_DVC_train_path,"\","gt"); % groundtruth image
    DVC_train=fullfile(Rose_1_DVC_train_img_path,img_filetype);
    DVC_train=dir(DVC_train);
    DVC_gt=fullfile(Rose_1_DVC_train_gt_path,gt_filetype);
    DVC_gt=dir(DVC_gt);
    % valid
    Rose_1_DVC_valid_path=strcat(Rose_1_DVC_path,"\","valid");
    Rose_1_DVC_valid_img_path=strcat(Rose_1_DVC_valid_path,"\","img"); % raw image
    Rose_1_DVC_valid_gt_path=strcat(Rose_1_DVC_valid_path,"\","gt"); % groundtruth image
    DVC_valid=fullfile(Rose_1_DVC_valid_img_path,img_filetype);
    DVC_valid=dir(DVC_valid);
    DVC_valid_gt=fullfile(Rose_1_DVC_valid_gt_path,gt_filetype);
    DVC_valid_gt=dir(DVC_valid_gt);
    % testing
    Rose_1_DVC_test_path=strcat(Rose_1_DVC_path,"\","test");
    Rose_1_DVC_test_img_path=strcat(Rose_1_DVC_test_path,"\","img"); % raw image
    Rose_1_DVC_test_gt_path=strcat(Rose_1_DVC_test_path,"\","gt"); % groundtruth image
    DVC_test=fullfile(Rose_1_DVC_test_img_path,img_filetype);
    DVC_test=dir(DVC_test);
    DVC_test_gt=fullfile(Rose_1_DVC_test_gt_path,gt_filetype);
    DVC_test_gt=dir(DVC_test_gt);
    % store each image (training)
    example_path=fullfile(Rose_1_DVC_train_img_path,DVC_train(1).name);
    example_path_gt=fullfile(Rose_1_DVC_train_gt_path,DVC_gt(1).name);
    example_file=imread(example_path);
    example_file_gt=imread(example_path_gt);
    num_files=length(DVC_train);
    [row,col,~]=size(example_file);
    type=class(example_file);
    type_gt=class(example_file_gt);
    train_image_storage=zeros(row,col,num_files,type);
    gt_image_storage=zeros(row,col,num_files,type_gt);
    % store each image (training)
    for i_file=1:length(DVC_train)
        current_train=DVC_train(i_file).name;
        current_gt=DVC_gt(i_file).name;
        full_current_train=fullfile(Rose_1_DVC_train_img_path,current_train);
        full_current_gt=fullfile(Rose_1_DVC_train_gt_path,current_gt);
        current_train_image=imread(full_current_train);
        current_train_gt=imread(full_current_gt);
        train_image_storage(:,:,i_file)=current_train_image;
        gt_image_storage(:,:,i_file)=current_train_gt;
    end
    % store each image (valid)
    num_valid_files=length(DVC_valid);
    valid_image_storage=zeros(row,col,num_files,type);
    valid_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_valid_files
        current_valid=DVC_valid(i_file).name;
        current_valid_gt=DVC_valid_gt(i_file).name;
        full_current_valid=fullfile(Rose_1_DVC_valid_img_path,current_valid);
        full_current_valid_gt=fullfile(Rose_1_DVC_valid_gt_path,current_valid_gt);
        current_valid_image=imread(full_current_valid);
        current_valid_gt=imread(full_current_valid_gt);
        valid_image_storage(:,:,i_file)=current_valid_image;
        valid_gt_image_storage(:,:,i_file)=current_valid_gt;
    end
    % store each image (test)
    num_test_files=length(DVC_test);
    test_image_storage=zeros(row,col,num_files,type);
    test_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_test_files
        current_test=DVC_test(i_file).name;
        current_test_gt=DVC_test_gt(i_file).name;
        full_current_test=fullfile(Rose_1_DVC_test_img_path,current_test);
        full_current_test_gt=fullfile(Rose_1_DVC_test_gt_path,current_test_gt);
        current_test_image=imread(full_current_test);
        current_test_gt=imread(full_current_test_gt);
        test_image_storage(:,:,i_file)=current_test_image;
        test_gt_image_storage(:,:,i_file)=current_test_gt;
    end
    % save each file
    save("processed_mat\DVC_train_img.mat","train_image_storage")
    save("processed_mat\DVC_train_gt.mat","gt_image_storage")
    save("processed_mat\DVC_valid_img.mat","valid_image_storage")
    save("processed_mat\DVC_valid_gt.mat","valid_gt_image_storage")
    save("processed_mat\DVC_test_img.mat","test_image_storage")
    save("processed_mat\DVC_test_gt.mat","test_gt_image_storage")
    % SVC
    Rose_1_SVC_path=strcat(Rose_1_path,"\","SVC");
    % training
    img_filetype='*.tif';
    gt_filetype='*.tif';
    thickgt_filetype='*.tif';
    thingt_filetype='*.tif';
    Rose_1_SVC_train_path=strcat(Rose_1_SVC_path,"\","train");
    Rose_1_SVC_train_img_path=strcat(Rose_1_SVC_train_path,"\","img"); % raw image
    Rose_1_SVC_train_gt_path=strcat(Rose_1_SVC_train_path,"\","gt"); % groundtruth image
    Rose_1_SVC_train_thickgt_path=strcat(Rose_1_SVC_train_path,"\","thick_gt"); % thick groundtruth image
    Rose_1_SVC_train_thingt_path=strcat(Rose_1_SVC_train_path,"\","thin_gt"); % thin groundtruth image
    SVC_train=fullfile(Rose_1_SVC_train_img_path,img_filetype);
    SVC_train=dir(SVC_train);
    SVC_gt=fullfile(Rose_1_SVC_train_gt_path,gt_filetype);
    SVC_gt=dir(SVC_gt);
    SVC_thickgt=fullfile(Rose_1_SVC_train_thickgt_path,thickgt_filetype);
    SVC_thickgt=dir(SVC_thickgt);
    SVC_thingt=fullfile(Rose_1_SVC_train_thingt_path,thingt_filetype);
    SVC_thingt=dir(SVC_thingt);
    % valid
    Rose_1_SVC_valid_path=strcat(Rose_1_SVC_path,"\","valid");
    Rose_1_SVC_valid_img_path=strcat(Rose_1_SVC_valid_path,"\","img"); % raw image
    Rose_1_SVC_valid_gt_path=strcat(Rose_1_SVC_valid_path,"\","gt"); % groundtruth image
    Rose_1_SVC_valid_thickgt_path=strcat(Rose_1_SVC_valid_path,"\","thick_gt"); % thick groundtruth image
    Rose_1_SVC_valid_thingt_path=strcat(Rose_1_SVC_valid_path,"\","thin_gt"); % thin groundtruth image
    SVC_valid=fullfile(Rose_1_SVC_valid_img_path,img_filetype);
    SVC_valid=dir(SVC_valid);
    SVC_valid_gt=fullfile(Rose_1_SVC_valid_gt_path,gt_filetype);
    SVC_valid_gt=dir(SVC_valid_gt);
    SVC_valid_thickgt=fullfile(Rose_1_SVC_valid_thickgt_path,thickgt_filetype);
    SVC_valid_thickgt=dir(SVC_valid_thickgt);
    SVC_valid_thingt=fullfile(Rose_1_SVC_valid_thingt_path,thingt_filetype);
    SVC_valid_thingt=dir(SVC_valid_thingt);
    % testing
    Rose_1_SVC_test_path=strcat(Rose_1_SVC_path,"\","test");
    Rose_1_SVC_test_img_path=strcat(Rose_1_SVC_test_path,"\","img"); % raw image
    Rose_1_SVC_test_gt_path=strcat(Rose_1_SVC_test_path,"\","gt"); % groundtruth image
    Rose_1_SVC_test_thickgt_path=strcat(Rose_1_SVC_test_path,"\","thick_gt"); % thick groundtruth image
    Rose_1_SVC_test_thingt_path=strcat(Rose_1_SVC_test_path,"\","thin_gt"); % thin groundtruth image
    SVC_test=fullfile(Rose_1_SVC_test_img_path,img_filetype);
    SVC_test=dir(SVC_test);
    SVC_test_gt=fullfile(Rose_1_SVC_test_gt_path,gt_filetype);
    SVC_test_gt=dir(SVC_test_gt);
    SVC_test_thickgt=fullfile(Rose_1_SVC_test_thickgt_path,thickgt_filetype);
    SVC_test_thickgt=dir(SVC_test_thickgt);
    SVC_test_thingt=fullfile(Rose_1_SVC_test_thingt_path,thingt_filetype);
    SVC_test_thingt=dir(SVC_test_thingt);
    % store each image (training)
    example_path=fullfile(Rose_1_SVC_train_img_path,SVC_train(1).name);
    example_path_gt=fullfile(Rose_1_SVC_train_gt_path,SVC_gt(1).name);
    example_path_thickgt=fullfile(Rose_1_SVC_train_thickgt_path,SVC_thickgt(1).name);
    example_path_thingt=fullfile(Rose_1_SVC_train_thingt_path,SVC_thingt(1).name);
    example_file=imread(example_path);
    example_file_gt=imread(example_path_gt);
    example_file_thickgt=imread(example_path_thickgt);
    example_file_thingt=imread(example_path_thingt);
    num_files=length(SVC_train);
    [row,col,~]=size(example_file);
    type=class(example_file);
    type_gt=class(example_file_gt);
    type_thickgt=class(example_file_thickgt);
    type_thingt=class(example_file_thingt);
    train_image_storage=zeros(row,col,num_files,type);
    gt_image_storage=zeros(row,col,num_files,type_gt);
    thickgt_image_storage=zeros(row,col,num_files,type_thickgt);
    thingt_image_storage=zeros(row,col,num_files,type_thingt);
    % store each image (training)
    for i_file=1:length(SVC_train)
        current_train=SVC_train(i_file).name;
        current_gt=SVC_gt(i_file).name;
        current_thickgt=SVC_thickgt(i_file).name;
        current_thingt=SVC_thingt(i_file).name;
        full_current_train=fullfile(Rose_1_SVC_train_img_path,current_train);
        full_current_gt=fullfile(Rose_1_SVC_train_gt_path,current_gt);
        full_current_thickgt=fullfile(Rose_1_SVC_train_thickgt_path,current_thickgt);
        full_current_thingt=fullfile(Rose_1_SVC_train_thingt_path,current_thingt);
        current_train_image=imread(full_current_train);
        current_train_gt=imread(full_current_gt);
        current_train_thickgt=imread(full_current_thickgt);
        current_train_thingt=imread(full_current_thingt);
        train_image_storage(:,:,i_file)=current_train_image;
        gt_image_storage(:,:,i_file)=current_train_gt;
        thickgt_image_storage(:,:,i_file)=current_train_thickgt;
        thingt_image_storage(:,:,i_file)=current_train_thingt;
    end
    % store each image (valid)
    num_valid_files=length(SVC_valid);
    valid_image_storage=zeros(row,col,num_files,type);
    valid_gt_image_storage=zeros(row,col,num_files,type_gt);
    valid_thickgt_image_storage=zeros(row,col,num_files,type_thickgt);
    valid_thingt_image_storage=zeros(row,col,num_files,type_thingt);
    for i_file=1:num_valid_files
        current_valid=SVC_valid(i_file).name;
        current_valid_gt=SVC_valid_gt(i_file).name;
        current_valid_thickgt=SVC_valid_thickgt(i_file).name;
        current_valid_thingt=SVC_valid_thingt(i_file).name;
        full_current_valid=fullfile(Rose_1_SVC_valid_img_path,current_valid);
        full_current_valid_gt=fullfile(Rose_1_SVC_valid_gt_path,current_valid_gt);
        full_current_valid_thickgt=fullfile(Rose_1_SVC_valid_thickgt_path,current_valid_thickgt);
        full_current_valid_thingt=fullfile(Rose_1_SVC_valid_thingt_path,current_valid_thingt);
        current_valid_image=imread(full_current_valid);
        current_valid_gt=imread(full_current_valid_gt);
        current_valid_thickgt=imread(full_current_valid_thickgt);
        current_valid_thingt=imread(full_current_valid_thingt);
        valid_image_storage(:,:,i_file)=current_valid_image;
        valid_gt_image_storage(:,:,i_file)=current_valid_gt;
        valid_thickgt_image_storage(:,:,i_file)=current_valid_thickgt;
        valid_thingt_image_storage(:,:,i_file)=current_valid_thingt;
    end
    % store each image (test)
    num_test_files=length(SVC_test);
    test_image_storage=zeros(row,col,num_files,type);
    test_gt_image_storage=zeros(row,col,num_files,type_gt);
    test_thickgt_image_storage=zeros(row,col,num_files,type_thickgt);
    test_thingt_image_storage=zeros(row,col,num_files,type_thingt);
    for i_file=1:num_test_files
        current_test=SVC_test(i_file).name;
        current_test_gt=SVC_test_gt(i_file).name;
        current_test_thickgt=SVC_test_thickgt(i_file).name;
        current_test_thingt=SVC_test_thingt(i_file).name;
        full_current_test=fullfile(Rose_1_SVC_test_img_path,current_test);
        full_current_test_gt=fullfile(Rose_1_SVC_test_gt_path,current_test_gt);
        full_current_test_thickgt=fullfile(Rose_1_SVC_test_thickgt_path,current_test_thickgt);
        full_current_test_thingt=fullfile(Rose_1_SVC_test_thingt_path,current_test_thingt);
        current_test_image=imread(full_current_test);
        current_test_gt=imread(full_current_test_gt);
        current_test_thickgt=imread(full_current_test_thickgt);
        current_test_thingt=imread(full_current_test_thingt);
        test_image_storage(:,:,i_file)=current_test_image;
        test_gt_image_storage(:,:,i_file)=current_test_gt;
        test_thickgt_image_storage(:,:,i_file)=current_test_thickgt;
        test_thingt_image_storage(:,:,i_file)=current_test_thingt;
    end
    % save each file
    save("processed_mat\SVC_train_img.mat","train_image_storage")
    save("processed_mat\SVC_train_gt.mat","gt_image_storage")
    save("processed_mat\SVC_train_thickgt.mat","thickgt_image_storage")
    save("processed_mat\SVC_train_thingt.mat","thingt_image_storage")
    save("processed_mat\SVC_valid_img.mat","valid_image_storage")
    save("processed_mat\SVC_valid_gt.mat","valid_gt_image_storage")
    save("processed_mat\SVC_valid_thickgt.mat","valid_thickgt_image_storage")
    save("processed_mat\SVC_valid_thingt.mat","valid_thingt_image_storage")
    save("processed_mat\SVC_test_img.mat","test_image_storage")
    save("processed_mat\SVC_test_gt.mat","test_gt_image_storage")
    save("processed_mat\SVC_test_thickgt.mat","test_thickgt_image_storage")
    save("processed_mat\SVC_test_thingt.mat","test_thingt_image_storage")

    % Rose-2
    Rose_2_path=strcat(Rose_path,"\","ROSE-2");
    % training
    img_filetype='*.png';
    gt_filetype='*.png';
    Rose_2_train_path=strcat(Rose_2_path,"\","train");
    Rose_2_train_img_path=strcat(Rose_2_train_path,"\","img"); % raw image
    Rose_2_train_gt_path=strcat(Rose_2_train_path,"\","gt"); % groundtruth image
    Rose_2_train=fullfile(Rose_2_train_img_path,img_filetype);
    Rose_2_train=dir(Rose_2_train);
    Rose_2_gt=fullfile(Rose_2_train_gt_path,gt_filetype);
    Rose_2_gt=dir(Rose_2_gt);
    % valid
    Rose_2_valid_path=strcat(Rose_2_path,"\","valid");
    Rose_2_valid_img_path=strcat(Rose_2_valid_path,"\","img"); % raw image
    Rose_2_valid_gt_path=strcat(Rose_2_valid_path,"\","gt"); % groundtruth image
    Rose_2_valid=fullfile(Rose_2_valid_img_path,img_filetype);
    Rose_2_valid=dir(Rose_2_valid);
    Rose_2_valid_gt=fullfile(Rose_2_valid_gt_path,gt_filetype);
    Rose_2_valid_gt=dir(Rose_2_valid_gt);
    % testing
    Rose_2_test_path=strcat(Rose_2_path,"\","test");
    Rose_2_test_img_path=strcat(Rose_2_test_path,"\","img"); % raw image
    Rose_2_test_gt_path=strcat(Rose_2_test_path,"\","gt"); % groundtruth image
    Rose_2_test=fullfile(Rose_2_test_img_path,img_filetype);
    Rose_2_test=dir(Rose_2_test);
    Rose_2_test_gt=fullfile(Rose_2_test_gt_path,gt_filetype);
    Rose_2_test_gt=dir(Rose_2_test_gt);
    % store each image (training)
    example_path=fullfile(Rose_2_train_img_path,Rose_2_train(1).name);
    example_path_gt=fullfile(Rose_2_train_gt_path,Rose_2_gt(1).name);
    example_file=imread(example_path);
    example_file_gt=imread(example_path_gt);
    num_files=length(Rose_2_train);
    [row,col,~]=size(example_file);
    type=class(example_file);
    type_gt=class(example_file_gt);
    train_image_storage=zeros(row,col,num_files,type);
    gt_image_storage=zeros(row,col,num_files,type_gt);
    % store each image (training)
    for i_file=1:num_files
        current_train=Rose_2_train(i_file).name;
        current_gt=Rose_2_gt(i_file).name;
        full_current_train=fullfile(Rose_2_train_img_path,current_train);
        full_current_gt=fullfile(Rose_2_train_gt_path,current_gt);
        current_train_image=imread(full_current_train);
        current_train_gt=imread(full_current_gt);
        train_image_storage(:,:,i_file)=current_train_image;
        gt_image_storage(:,:,i_file)=current_train_gt;
    end
    % store each image (valid)
    num_valid_files=length(Rose_2_valid);
    valid_image_storage=zeros(row,col,num_files,type);
    valid_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_valid_files
        current_valid=Rose_2_valid(i_file).name;
        current_valid_gt=Rose_2_valid_gt(i_file).name;
        full_current_valid=fullfile(Rose_2_valid_img_path,current_valid);
        full_current_valid_gt=fullfile(Rose_2_valid_gt_path,current_valid_gt);
        current_valid_image=imread(full_current_valid);
        current_valid_gt=imread(full_current_valid_gt);
        valid_image_storage(:,:,i_file)=current_valid_image;
        valid_gt_image_storage(:,:,i_file)=current_valid_gt;
    end
    % store each image (test)
    num_test_files=length(Rose_2_test);
    test_image_storage=zeros(row,col,num_files,type);
    test_gt_image_storage=zeros(row,col,num_files,type_gt);
    for i_file=1:num_test_files
        current_test=Rose_2_test(i_file).name;
        current_test_gt=Rose_2_test_gt(i_file).name;
        full_current_test=fullfile(Rose_2_test_img_path,current_test);
        full_current_test_gt=fullfile(Rose_2_test_gt_path,current_test_gt);
        current_test_image=imread(full_current_test);
        current_test_gt=imread(full_current_test_gt);
        test_image_storage(:,:,i_file)=current_test_image;
        test_gt_image_storage(:,:,i_file)=current_test_gt;
    end
    % save each file
    save("processed_mat\rose_2_train_img.mat","train_image_storage")
    save("processed_mat\rose_2_train_gt.mat","gt_image_storage")
    save("processed_mat\rose_2_valid_img.mat","valid_image_storage")
    save("processed_mat\rose_2_valid_gt.mat","valid_gt_image_storage")
    save("processed_mat\rose_2_test_img.mat","test_image_storage")
    save("processed_mat\rose_2_test_gt.mat","test_gt_image_storage")



end