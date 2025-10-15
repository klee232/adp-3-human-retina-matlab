% Created by Kuan-Min Lee
% Created date: Jan. 10th, 2024 (last updated: Jan. 15th, 2024)
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This script is created to initialize the data preprocessing of ROSE
% dataset

% Input Parameter: None
% Output Parameter:
%
% SVC dataset:
% train_ROSE_SVC_org: Training image for SVC (multi-dimensional image array)
% valid_ROSE_SVC_org: Validation image for SVC (multi-dimensional image array)
% train_ROSE_SVC_orgGt: Ground truth for training image for SVC
% (multi-dimensional image array)
% valid_ROSE_SVC_ orgGt: Ground truth for validation image for SVC
% (multi-dimensional image array)
% train_ROSE_SVC_thinGt: Thin vessel Ground truth for training image for SVC
% (multi-dimensional image array)
% valid_ROSE_SVC_thinGt: Thin vessel Ground truth for validation image for SVC
% (multi-dimensional image array)
% train_ROSE_SVC_thickGt: Thick vessel Ground truth for training image for SVC 
% (multi-dimensional image array)
% valid_ROSE_SVC_thickGt: Thick vessel Ground truth for validation image
% for SVC
% (multi-dimensional image array)
% test_ROSE_SVC_org: Testing image for SVC (multi-dimensional image array)
% test_ROSE_SVC_orgGt: Testing Ground Truth image for SVC (imageDataStore
% object)
% test_ROSE_SVC_thinGt: Thin vessel Ground Truth image for SVC
% (multi-dimensional image array)
% test_ROSE_SVC_thickGt: Thick vessel Ground Truth image for SVC
% (multi-dimensional image array)
%
%
% DVC dataset:
% train_ROSE_DVC_org: Training image for DVC (multi-dimensional image array)
% valid_ROSE_DVC_org: Validation image for DVC (multi-dimensional image array)
% train_ROSE_DVC_orgGt: Ground truth for training image for DVC
% (multi-dimensional image array)
% valid_ROSE_DVC_ orgGt: Ground truth for validation image for DVC
% (multi-dimensional image array)
% test_ROSE_DVC_org: Testing image for DVC (multi-dimensional image array)
% test_ROSE_DVC_orgGt: Testing Ground Truth image for DVC (multi-dimensional image array)
%
%
% SVC_DVC dataset:
% train_ROSE_SDVC_org: Training image for SDVC (multi-dimensional image array)
% valid_ROSE_SDVC_org: Validation image for SDVC (multi-dimensional image array)
% train_ROSE_SDVC_orgGt: Ground truth for training image for SDVC
% (multi-dimensional image array)
% valid_ROSE_SDVC_ orgGt: Ground truth for validation image for SDVC
% (multi-dimensional image array)
% test_ROSE_SDVC_org: Testing image for SDVC (multi-dimensional image array)
% test_ROSE_SDVC_orgGt: Testing Ground Truth image for SDVC (imageDataStore
% object)

function [train_ROSE_SVC_org, valid_ROSE_SVC_org, ...
          train_ROSE_SVC_orgGt, valid_ROSE_SVC_orgGt, ...
          train_ROSE_SVC_thinGt, valid_ROSE_SVC_thinGt, ...
          train_ROSE_SVC_thickGt, valid_ROSE_SVC_thickGt, ...
          test_ROSE_SVC_org, test_ROSE_SVC_orgGt, test_ROSE_SVC_thinGt, test_ROSE_SVC_thickGt,...
          train_ROSE_DVC_org, valid_ROSE_DVC_org, ...
          train_ROSE_DVC_orgGt, valid_ROSE_DVC_orgGt, ...
          test_ROSE_DVC_org, test_ROSE_DVC_orgGt, ...
          train_ROSE_SDVC_org, valid_ROSE_SDVC_org, ...
          train_ROSE_SDVC_orgGt, valid_ROSE_SDVC_orgGt, ...
          test_ROSE_SDVC_org, test_ROSE_SDVC_orgGt]=data_enface_ROSE_data_initiater()

    %% display loading information (checked and updated)
    disp("Start ROSE-1 Data Preprocessing...")
    addpath(genpath("data preprocessing function\"))


    %% Pathway setup (checked and updated)
    data_path_rose="~/data/klee232/2D dataset/ROSE/data/ROSE-1";

    % Valid data ratio
    valid_ratio=0.15;
 

    %% loading dataset (checked and updated)
    %% SVC dataset
    filetype="tif";
    % sub pathway setup
    train_data_path_rose_SVC_org=strcat(data_path_rose,"/SVC/train/img");
    train_data_path_rose_SVC_orgGt=strcat(data_path_rose,"/SVC/train/gt");
    train_data_path_rose_SVC_thinGt=strcat(data_path_rose,"/SVC/train/thin_gt");
    train_data_path_rose_SVC_thickGt=strcat(data_path_rose,"/SVC/train/thick_gt");
    test_data_path_rose_SVC_org=strcat(data_path_rose,"/SVC/test/img");
    test_data_path_rose_SVC_orgGt=strcat(data_path_rose,"/SVC/test/gt");
    test_data_path_rose_SVC_thinGt=strcat(data_path_rose,"/SVC/test/thin_gt");
    test_data_path_rose_SVC_thickGt=strcat(data_path_rose,"/SVC/test/thick_gt");

    % for training dataset and validation
    [train_ROSE_SVC_org,valid_ROSE_SVC_org]=image_rose_data_ROSE_process(train_data_path_rose_SVC_org,filetype,valid_ratio);
    [train_ROSE_SVC_orgGt,valid_ROSE_SVC_orgGt]=image_rose_data_ROSE_process(train_data_path_rose_SVC_orgGt,filetype,valid_ratio);
    [train_ROSE_SVC_thinGt,valid_ROSE_SVC_thinGt]=image_rose_data_ROSE_process(train_data_path_rose_SVC_thinGt,filetype,valid_ratio);
    [train_ROSE_SVC_thickGt,valid_ROSE_SVC_thickGt]=image_rose_data_ROSE_process(train_data_path_rose_SVC_thickGt,filetype,valid_ratio);

    % for testing dataset
    [test_ROSE_SVC_org]=image_rose_data_ROSE_test_process(test_data_path_rose_SVC_org,filetype);
    [test_ROSE_SVC_orgGt]=image_rose_data_ROSE_test_process(test_data_path_rose_SVC_orgGt,filetype);
    [test_ROSE_SVC_thinGt]=image_rose_data_ROSE_test_process(test_data_path_rose_SVC_thinGt,filetype);
    [test_ROSE_SVC_thickGt]=image_rose_data_ROSE_test_process(test_data_path_rose_SVC_thickGt,filetype);

    %% DVC dataset
    % sub pathway setup
    filetype="tif";
    train_data_path_rose_DVC_org=strcat(data_path_rose,"/DVC/train/img");
    train_data_path_rose_DVC_orgGt=strcat(data_path_rose,"/DVC/train/gt");
    test_data_path_rose_DVC_org=strcat(data_path_rose,"/DVC/test/img");
    test_data_path_rose_DVC_orgGt=strcat(data_path_rose,"/DVC/test/gt");
    % loading dataset
    % for training dataset
    [train_ROSE_DVC_org,valid_ROSE_DVC_org]=image_rose_data_ROSE_process(train_data_path_rose_DVC_org,filetype,valid_ratio);
    [train_ROSE_DVC_orgGt,valid_ROSE_DVC_orgGt]=image_rose_data_ROSE_process(train_data_path_rose_DVC_orgGt,filetype,valid_ratio);

    % for testing dataset
    [test_ROSE_DVC_org]=image_rose_data_ROSE_test_process(test_data_path_rose_DVC_org,filetype);
    [test_ROSE_DVC_orgGt]=image_rose_data_ROSE_test_process(test_data_path_rose_DVC_orgGt,filetype);
    
    %% SVC + DVC dataset 
    % sub pathway setup
    filetype="png";
    filetype2="tif";
    train_data_path_rose_SDVC_org=strcat(data_path_rose,"/SVC_DVC/train/img");
    train_data_path_rose_SDVC_orgGt=strcat(data_path_rose,"/SVC_DVC/train/gt");
    test_data_path_rose_SDVC_org=strcat(data_path_rose,"/SVC_DVC/test/img");
    test_data_path_rose_SDVC_orgGt=strcat(data_path_rose,"/SVC_DVC/test/gt");
    % loading dataset
    % for training dataset
    [train_ROSE_SDVC_org,valid_ROSE_SDVC_org]=image_rose_data_ROSE_process(train_data_path_rose_SDVC_org,filetype,valid_ratio);
    [train_ROSE_SDVC_orgGt,valid_ROSE_SDVC_orgGt]=image_rose_data_ROSE_process(train_data_path_rose_SDVC_orgGt,filetype2,valid_ratio);

    % for testing dataset
    [test_ROSE_SDVC_org]=image_rose_data_ROSE_test_process(test_data_path_rose_SDVC_org,filetype);
    [test_ROSE_SDVC_orgGt]=image_rose_data_ROSE_test_process(test_data_path_rose_SDVC_orgGt,filetype2);
    
    disp("ROSE-1 Data Preprocessing Ends")

end