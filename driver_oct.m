%% Phase 1 Segmentation
% This part contains the groundtruth generation for octa image.
% Inside this part, there are two sub processes:
% 1. data_processor: process for generating groundtruth and grab out octa
% image
% 2. data_unifer: process for unifying the input data size for later nerual
% network input
clear all

addpath(genpath("Phase_1_Segmentation/data function/"))
addpath(genpath("Phase_1_Segmentation/data refiner function/"))

% if the current oct data existed, load the images and skip current
% processing
if exist('~/data/klee232/processed_data/oct_data.mat')
    if ~exist('~/data/klee232/processed_data/pad_octa_data_frangi.mat')
        load('~/data/klee232/processed_data/oct_data.mat');
        load('~/data/klee232/processed_data/oct_seg_data.mat');
    end
% if not, conduct data processing 
else
    % process the dataset
    [oct_storage,oct_seg_storage]=data_processor_oct();
end

% united dataset size for neural network
% again, check if the process is already done
% if so, load the data and skip this part
if exist('~/data/klee232/processed_data/pad_oct_data.mat')
    load('~/data/klee232/processed_data/pad_oct_data.mat');
    load('~/data/klee232/processed_data/pad_oct_seg_data.mat');
% if not, conduct data size unifer
else
    [oct_storage,oct_seg_storage]=data_unifier(oct_storage,oct_seg_storage);
end

% keep only surface, deep and choroid layer
[oct_storage,oct_seg_storage]=data_filter_oct(oct_storage,oct_seg_storage);



%% Phase 2 Input Data Process (more in progress)
% this part serves as an additional process for input image for neural
% network. 
% planned task:
% 1. octa region sorting: a function that is able to sort the octa image
% into retinal quardants

addpath(genpath("Phase_2_Input_Data_Process/"))

% create labels for prediction model
[label_gt_storage]=data_labelCreator();

% create folded data
Num_fold=4;
Test_ratio=0.25;
[train_oct_storage, valid_oct_storage, test_oct_storage,...
 train_oct_seg_storage, valid_oct_seg_storage, test_oct_seg_storage,...
 train_label_gt_storage, valid_label_gt_storage, test_label_gt_storage]=data_partitioner(oct_storage,oct_seg_storage,label_gt_storage,Num_fold,Test_ratio);


%% Phase 3 launch the neural network
% this part serves as the neural network training and testing parts for our
% classification neural network
addpath(genpath("Phase_3_Prediction_model/"))

train_oct_seg_storage=single(train_oct_seg_storage);
valid_oct_seg_storage=single(valid_oct_seg_storage);

% launch the neural network
classes=unique(train_label_gt_storage);
num_classes=size(classes,1);
[img_depth, img_row, img_col,~]=size(train_oct_storage);
network=original_network_mean_removal_pca_net_launcher(img_depth, img_row, img_col, num_classes);

plot(network)

% start the neural network training
num_epoch=200;
learning_rate=0.01;
trained_original_network_mean_removal_pca_oct_incomplete=training_function(network,train_oct_seg_storage,train_label_gt_storage,...
                                                                 valid_oct_seg_storage,valid_label_gt_storage,...
                                                                 num_epoch,learning_rate);

save Phase_3_Prediction_model/models/main_model/trained_original_network_mean_removal_pca_oct_incomplete
load("Phase_3_Prediction_model/models/main_model/trained_original_network_mean_removal_pca_oct_incomplete.mat")


% start the neural network testing
[test_loss,confusion_matrix]=testing_function(trained_original_network_mean_removal_pca,test_octa_gt_storage,test_label_gt_storage,Num_fold);


%% Phase 4 Testing Outcome Visualization

