% Created by Kuan-Min Lee
% Created date: Dec. 11th, 2023
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This script file is created to initiate the dataset from ROSE-2 dataset

% Input parameter: None
%
% dataset:
% train_ROSE2_org: Training image (imageDataStore object)
% valid_ROSE2_org: Validation image (imageDataStore object)
% train_ROSE2_orgGt: Ground truth for training image
% (imageDataStore object)
% valid_ROSE2_orgGt: Ground truth for validation image
% (imageDataStore object)
% test_ROSE2_org: Testing image (imageDataStore object)
% test_ROSE2_orgGt: Testing Ground Truth image (imageDataStore
% object)

function [train_ROSE2_org, valid_ROSE2_org, ...
          train_ROSE2_orgGt, valid_ROSE2_orgGt, ...
          test_ROSE2_org, test_ROSE2_orgGt]=data_enface_ROSE2_data_initiater()

    %% display loading information (checked and updated)
    disp("Start ROSE-2 Data Preprocessing...")


    %% Pathway setup (checked and updated)
    data_path_rose2="~/data/klee232/2D dataset/ROSE/data/ROSE-2";

    % Valid data ratio
    valid_ratio=0.15;


    %% training and testing dataset pathway setup
    train_data_path_rose2_org=strcat(data_path_rose2,"/train/original");
    train_data_path_rose2_orgGt=strcat(data_path_rose2,"/train/gt");
    test_data_path_rose2_org=strcat(data_path_rose2,"/test/original");
    test_data_path_rose2_orgGt=strcat(data_path_rose2,"/test/gt");


    %% loading dataset
    filetype="png";
    [train_ROSE2_org,valid_ROSE2_org]=image_rose_data_ROSE_process(train_data_path_rose2_org,filetype,valid_ratio);
    [train_ROSE2_orgGt,valid_ROSE2_orgGt]=image_rose_data_ROSE_process(train_data_path_rose2_orgGt,filetype,valid_ratio);

    % for testing dataset
    [test_ROSE2_org]=image_rose_data_ROSE_test_process(test_data_path_rose2_org,filetype);
    [test_ROSE2_orgGt]=image_rose_data_ROSE_test_process(test_data_path_rose2_orgGt,filetype);

    disp("ROSE-2 Data Loading ends /n")

end