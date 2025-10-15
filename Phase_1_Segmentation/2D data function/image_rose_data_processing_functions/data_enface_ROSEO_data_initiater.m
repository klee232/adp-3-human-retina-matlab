% Created by Kuan-Min Lee
% Created date: Jan. 10th, 2024 (last updated: Jan. 10th, 2024)
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This script is created to initialize the data preprocessing of ROSE
% dataset

% Input Parameter: None
% Output Parameter:
%
% SVC dataset:
% train_ROSEO_SVC_org: Training image for SVC (imageDataStore object)
% valid_ROSEO_SVC_org: Validation image for SVC (imageDataStore object)
% train_ROSEO_SVC_orgGt: Ground truth for training image for SVC
% (imageDataStore object)
% valid_ROSEO_SVC_orgGt: Ground truth for validation image for SVC
% (imageDataStore object)
% test_ROSEO_SVC_org: Testing image for SVC (imageDataStore object)
% test_ROSEO_SVC_orgGt: Testing Ground Truth image for SVC (imageDataStore
% object)
%
%
% DVC dataset:
% train_ROSEO_DVC_org: Training image for DVC (imageDataStore object)
% valid_ROSEO_DVC_org: Validation image for DVC (imageDataStore object)
% train_ROSEO_DVC_orgGt: Ground truth for training image for DVC
% (imageDataStore object)
% valid_ROSEO_DVC_orgGt: Ground truth for validation image for DVC
% (imageDataStore object)
% test_ROSEO_DVC_org: Testing image for DVC (imageDataStore object)
% test_ROSEO_DVC_orgGt: Testing Ground Truth image for DVC (imageDataStore
% object)
%
%
% IVC dataset:
% train_ROSEO_IVC_org: Training image for IVC (imageDataStore object)
% valid_ROSEO_IVC_org: Validation image for IVC (imageDataStore object)
% train_ROSEO_IVC_orgGt: Ground truth for training image for IVC
% (imageDataStore object)
% valid_ROSEO_IVC_orgGt: Ground truth for validation image for IVC
% (imageDataStore object)
% test_ROSEO_IVC_org: Testing image for IVC (imageDataStore object)
% test_ROSEO_IVC_orgGt: Testing Ground Truth image for IVC (imageDataStore
% object)


function [train_ROSEO_SVC_org, valid_ROSEO_SVC_org, ...
          train_ROSEO_SVC_orgGt, valid_ROSEO_SVC_orgGt, ...
          test_ROSEO_SVC_org, test_ROSEO_SVC_orgGt, ...
          train_ROSEO_DVC_org, valid_ROSEO_DVC_org, ...
          train_ROSEO_DVC_orgGt, valid_ROSEO_DVC_orgGt, ...
          test_ROSEO_DVC_org, test_ROSEO_DVC_orgGt, ...
          train_ROSEO_IVC_org, valid_ROSEO_IVC_org, ...
          train_ROSEO_IVC_orgGt, valid_ROSEO_IVC_orgGt, ...
          test_ROSEO_IVC_org, test_ROSEO_IVC_orgGt]=ROSEO_data_initiater()

    disp("Start ROSE-O Data Preprocessing...")

    % Pathway setup
    data_path_roseo="C:\Users\klee232\Desktop\Thesis\Codes\My works\dataset\ROSE-O\ROSE-O";

    % Valid data ratio
    valid_ratio=0.85;

    % SVC dataset
    % sub pathway setup
    train_data_path_ROSEO_SVC_org=strcat(data_path_roseo,"\train\img\SVC");
    train_data_path_ROSEO_SVC_orgGt=strcat(data_path_roseo,"\train\gt\SVC");
    test_data_path_ROSEO_SVC_org=strcat(data_path_roseo,"\test\img\SVC");
    test_data_path_ROSEO_SVC_orgGt=strcat(data_path_roseo,"\test\gt\SVC");
    % loading dataset
    filetype="tif";
    % for training dataset
    [train_ROSEO_SVC_path,valid_ROSEO_SVC_path,~]=ROSE_process(train_data_path_ROSEO_SVC_org,filetype,valid_ratio);
    [train_ROSEO_SVC_gt_path,valid_ROSEO_SVC_gt_path,~]=ROSE_process(train_data_path_ROSEO_SVC_orgGt,filetype,valid_ratio);
    % for testing dataset
    [~,~,test_ROSEO_SVC_path]=ROSE_process(test_data_path_ROSEO_SVC_org,filetype,valid_ratio);
    [~,~,test_ROSEO_SVC_gt_path]=ROSE_process(test_data_path_ROSEO_SVC_orgGt,filetype,valid_ratio);
    % Save Images as ImageDatastore Object
    filetype=".tif";
    train_ROSEO_SVC_org=imageDatastore(strcat(train_ROSEO_SVC_path,"\img"),"FileExtensions",filetype);
    valid_ROSEO_SVC_org=imageDatastore(strcat(valid_ROSEO_SVC_path,"\img"),"FileExtensions",filetype);
    train_ROSEO_SVC_orgGt=imageDatastore(strcat(train_ROSEO_SVC_gt_path,"\gt"),"FileExtensions",filetype);
    valid_ROSEO_SVC_orgGt=imageDatastore(strcat(valid_ROSEO_SVC_gt_path,"\gt"),"FileExtensions",filetype);
    test_ROSEO_SVC_org=imageDatastore(strcat(test_ROSEO_SVC_path,"\img"),"FileExtensions",filetype);
    test_ROSEO_SVC_orgGt=imageDatastore(strcat(test_ROSEO_SVC_gt_path,"\gt"),"FileExtensions",filetype);

    % DVC dataset
    % sub pathway setup
    filetype="tif";
    train_data_path_ROSEO_DVC_org=strcat(data_path_roseo,"\DVC\train\img");
    train_data_path_ROSEO_DVC_orgGt=strcat(data_path_roseo,"\DVC\train\gt");
    test_data_path_ROSEO_DVC_org=strcat(data_path_roseo,"\DVC\test\img");
    test_data_path_ROSEO_DVC_orgGt=strcat(data_path_roseo,"\DVC\test\gt");
    % loading dataset
    % for training dataset
    [train_ROSEO_DVC_path,valid_ROSEO_DVC_path,~]=ROSE_process(train_data_path_ROSEO_DVC_org,filetype,valid_ratio);
    [train_ROSEO_DVC_gt_path,valid_ROSEO_DVC_gt_path,~]=ROSE_process(train_data_path_ROSEO_DVC_orgGt,filetype,valid_ratio);
    % for testing dataset
    [~,~,test_ROSEO_DVC_path]=ROSE_process(test_data_path_ROSEO_DVC_org,filetype,valid_ratio);
    [~,~,test_ROSEO_DVC_gt_path]=ROSE_process(test_data_path_ROSEO_DVC_orgGt,filetype,valid_ratio);
    % Save Images as ImageDatastore Object
    filetype=".tif";
    train_ROSEO_DVC_org=imageDatastore(strcat(train_ROSEO_DVC_path,"\img"),"FileExtensions",filetype);
    valid_ROSEO_DVC_org=imageDatastore(strcat(valid_ROSEO_DVC_path,"\img"),"FileExtensions",filetype);
    train_ROSEO_DVC_orgGt=imageDatastore(strcat(train_ROSEO_DVC_gt_path,"\gt"),"FileExtensions",filetype);
    valid_ROSEO_DVC_orgGt=imageDatastore(strcat(valid_ROSEO_DVC_gt_path,"\gt"),"FileExtensions",filetype);
    test_ROSEO_DVC_org=imageDatastore(strcat(test_ROSEO_DVC_path,"\img"),"FileExtensions",filetype);
    test_ROSEO_DVC_orgGt=imageDatastore(strcat(test_ROSEO_DVC_gt_path,"\gt"),"FileExtensions",filetype);

    % IVC dataset 
    % sub pathway setup
    filetype="png";
    train_data_path_ROSEO_IVC_org=strcat(data_path_roseo,"\IVC\train\img");
    train_data_path_ROSEO_IVC_orgGt=strcat(data_path_roseo,"\IVC\train\gt");
    test_data_path_ROSEO_IVC_org=strcat(data_path_roseo,"\IVC\test\img");
    test_data_path_ROSEO_IVC_orgGt=strcat(data_path_roseo,"\IVC\test\gt");
    % loading dataset
    % for training dataset
    [train_ROSEO_IVC_path,valid_ROSEO_IVC_path,~]=ROSE_process(train_data_path_ROSEO_IVC_org,filetype,valid_ratio);
    [train_ROSEO_IVC_gt_path,valid_ROSEO_IVC_gt_path,~]=ROSE_process(train_data_path_ROSEO_IVC_orgGt,filetype,valid_ratio);
    % for testing dataset
    [~,~,test_ROSEO_IVC_path]=ROSE_process(test_data_path_ROSEO_IVC_org,filetype,valid_ratio);
    [~,~,test_ROSEO_IVC_gt_path]=ROSE_process(test_data_path_ROSEO_IVC_orgGt,filetype,valid_ratio);
    % Save Images as ImageDatastore Object
    filetype=".png";
    train_ROSEO_IVC_org=imageDatastore(strcat(train_ROSEO_IVC_path,"\img"),"FileExtensions",filetype);
    valid_ROSEO_IVC_org=imageDatastore(strcat(valid_ROSEO_IVC_path,"\img"),"FileExtensions",filetype);
    train_ROSEO_IVC_orgGt=imageDatastore(strcat(train_ROSEO_IVC_gt_path,"\gt"),"FileExtensions",filetype);
    valid_ROSEO_IVC_orgGt=imageDatastore(strcat(valid_ROSEO_IVC_gt_path,"\gt"),"FileExtensions",filetype);
    test_ROSEO_IVC_org=imageDatastore(strcat(test_ROSEO_IVC_path,"\img"),"FileExtensions",filetype);
    test_ROSEO_IVC_orgGt=imageDatastore(strcat(test_ROSEO_IVC_gt_path,"\gt"),"FileExtensions",filetype);
    disp("ROSE-O Data Preprocessing Ends")

end