%% Phase 1 Segmentation
clear all

addpath(genpath("Phase_1_Segmentation/"))


%% 3D OCTA Processing
% This part contains the groundtruth generation for octa image.
% Inside this part, there are two sub processes:
% 1. data_processor: process for generating groundtruth and grab out octa
% image
% 2. data_unifer: process for unifying the input data size for later nerual
% network input

% % check if the data is already processed
% % if it's already processed, load the dataset directly and skip this part
% if exist('~/data/klee232/processed_data/octa gt arrays/octa_gt_data_complete_choroid_excluded_frangi.mat')
%     if ~exist('~/data/klee232/processed_data/octa arrays/pad_octa_data_complete_choroid_excluded_frangi.mat')
%         load('~/data/klee232/processed_data/octa arrays/octa_gt_data_complete_choroid_excluded_frangi.mat');
%         load('~/data/klee232/processed_data/octa gt arrays/octa_data_complete_choroid_excluded_frangi.mat');
%     end
% % if not, conduct data processing 
% else
%     % process the dataset
%     [octa_storage,octa_gt_storage]=data_processor();
% end

% % united dataset size for neural network
% % again, check if the process is already done
% % if so, load the data and skip this part
% if exist('~/data/klee232/processed_data/octa arrays/pad_octa_data_complete_choroid_excluded_frangi.mat')
%     if ~exist('~/data/klee232/processed_data/octa arrays/refined_pad_octa_data_complete_choroid_excluded_frangi.mat')
%         load('~/data/klee232/processed_data/octa arrays/pad_octa_data_complete_choroid_excluded_frangi.mat');
%         load('~/data/klee232/processed_data/octa gt arrays/pad_octa_gt_data_complete_choroid_excluded_frangi.mat');
%     end
% % if not, conduct data size unifer
% else
%     [octa_storage,octa_gt_storage]=data_unifier(octa_storage,octa_gt_storage);
% end

% % refine the stored image
% % if the image already existed, load the save file
% if exist('~/data/klee232/processed_data/refined_pad_octa_data_frangi.mat')
%     load('~/data/klee232/processed_data/pad_octa_data_frangi.mat');
%     load('~/data/klee232/processed_data/refined_pad_octa_data_frangi.mat');
% % if not, conduct image refining
% else
%     [denoised_octa_gt_storage]=image_groundtruth_refiner_denoise(octa_gt_storage);
% end


%% Phase 1: 2D OCTA enface Processing (Run this part if you want to input en-face feature
% This part contains the en_face construction for octa image and goundtruth
% generation for en face. Inside this part, there are two sub processes:
% 1. data_enface_creator: create enface image storage
% 2. data_enface_processor: create enface image segmentation

% execute en face creator or en face processor function if necessary
% if there exists individual en face layer
if exist("~/data/klee232/processed_data/octa en face arrays/octa_data_surface_en_face.mat") && ...
   exist("~/data/klee232/processed_data/octa en face arrays/octa_data_deep_en_face.mat") && ...
   exist("~/data/klee232/processed_data/octa en face arrays/octa_data_choroid_en_face.mat")
    % if there exists concatenated en face layer
    if exist("~/data/klee232/processed_data/octa en face arrays/octa_data_en_face.mat")
        % if there exists concatenated segmentation layer
        if exist("~/data/klee232/processed_data/octa en face arrays/octa_gt_data_en_face.mat")
           load("~/data/klee232/processed_data/octa en face arrays/octa_data_en_face.mat");
           load("~/data/klee232/processed_data/octa en face arrays/octa_gt_data_en_face.mat");
        % if there doesn't exist concatenated segmentation layer
        else
           load("~/data/klee232/processed_data/octa en face arrays/octa_data_en_face.mat");
           [en_face_octa_seg_storage]=data_enface_processor(); % create segmentation en face
        end
    % if there doesn't exist concatenated face layer
    else
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_surface_en_face.mat");
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_deep_en_face.mat");
        load("~/data/klee232/processed_data/octa en face arrays/octa_data_choroid_en_face.mat");
        [en_face_octa_storage]=data_en_face_concatenator(octa_surf_storage,octa_deep_storage,octa_choroid_storage); % create concatenated en face
        [en_face_octa_seg_storage]=data_enface_processor(); % create concatenated segmentation
    end
% if there exists no en face file
else
    [octa_surf_storage,octa_deep_storage,octa_choroid_storage]=data_enface_creator(); % create individual en face file
    [en_face_octa_storage]=data_en_face_concatenator(octa_surf_storage,octa_deep_storage,octa_choroid_storage); % create concatenated en face
    [en_face_octa_seg_storage]=data_enface_processor(); % create concatenated segmentation
end


%% Phase 1: load dataset
% ROSE dataset
[train_ROSE_SVC_org, valid_ROSE_SVC_org, ...
 train_ROSE_SVC_orgGt, valid_ROSE_SVC_orgGt, ...
 train_ROSE_SVC_thinGt, valid_ROSE_SVC_thinGt, ...
 train_ROSE_SVC_thickGt, valid_ROSE_SVC_thickGt, ...
 test_ROSE_SVC_org, test_ROSE_SVC_orgGt, test_ROSE_SVC_thinGt, test_ROSE_SVC_thickGt,...
 train_ROSE_DVC_org, valid_ROSE_DVC_org, ...
 train_ROSE_DVC_orgGt, valid_ROSE_DVC_orgGt, ...
 test_ROSE_DVC_org, test_ROSE_DVC_orgGt, ...
 train_ROSE_SDVC_org, valid_ROSE_SDVC_org, ...
 train_ROSE_SDVC_orgGt, valid_ROSE_SDVC_orgGt, ...
 test_ROSE_SDVC_org, test_ROSE_SDVC_orgGt]=data_enface_ROSE_data_initiater();

% ROSE-2 dataset
[train_ROSE2_org, valid_ROSE2_org, ...
 train_ROSE2_orgGt, valid_ROSE2_orgGt, ...
 test_ROSE2_org, test_ROSE2_orgGt]=data_enface_ROSE2_data_initiater();


%% Phase 1: create folded dataset
num_fold=4;
% Rose SVC dataset
[fold_train_rose_SVC_data_storage, fold_valid_rose_SVC_data_storage, ...
 fold_train_rose_SVC_gt_storage, fold_valid_rose_SVC_gt_storage,...
 fold_train_rose_SVC_thick_gt_storage, fold_valid_rose_SVC_thick_gt_storage,...
 fold_train_rose_SVC_thin_gt_storage, fold_valid_rose_SVC_thin_gt_storage]=data_en_face_ROSE_SVC_fold_creator(train_ROSE_SVC_org, train_ROSE_SVC_orgGt, train_ROSE_SVC_thickGt, train_ROSE_SVC_thinGt,...
                                                                                                              valid_ROSE_SVC_org, valid_ROSE_SVC_orgGt, valid_ROSE_SVC_thickGt, valid_ROSE_SVC_thinGt,num_fold);
fold_train_rose_SVC_gt_storage(find(fold_train_rose_SVC_gt_storage))=1;
fold_train_rose_SVC_thick_gt_storage(find(fold_train_rose_SVC_thick_gt_storage))=1;
fold_train_rose_SVC_thin_gt_storage(find(fold_train_rose_SVC_thin_gt_storage))=1;
fold_valid_rose_SVC_gt_storage(find(fold_valid_rose_SVC_gt_storage))=1;
fold_valid_rose_SVC_thick_gt_storage(find(fold_valid_rose_SVC_thick_gt_storage))=1;
fold_valid_rose_SVC_thin_gt_storage(find(fold_valid_rose_SVC_thin_gt_storage))=1;
test_rose_SVC_data_storage=test_ROSE_SVC_org;
test_rose_SVC_gt_storage=test_ROSE_SVC_orgGt;
test_rose_SVC_thick_gt_storage=test_ROSE_SVC_thickGt;
test_rose_SVC_thin_gt_storage=test_ROSE_SVC_thinGt;
test_rose_SVC_gt_storage(find(test_rose_SVC_gt_storage))=1;
test_rose_SVC_thick_gt_storage(find(test_rose_SVC_thick_gt_storage))=1;
test_rose_SVC_thin_gt_storage(find(test_rose_SVC_thin_gt_storage))=1;

% ROSE DVC dataset
[fold_train_rose_DVC_data_storage, fold_valid_rose_DVC_data_storage, ...
 fold_train_rose_DVC_gt_storage, fold_valid_rose_DVC_gt_storage]=data_en_face_ROSE_DVC_fold_creator(train_ROSE_DVC_org, train_ROSE_DVC_orgGt, valid_ROSE_DVC_org, valid_ROSE_DVC_orgGt,num_fold);
fold_train_rose_DVC_gt_storage(find(fold_train_rose_DVC_gt_storage))=1;
fold_valid_rose_DVC_gt_storage(find(fold_valid_rose_DVC_gt_storage))=1;
test_rose_DVC_data_storage=test_ROSE_DVC_org;
test_rose_DVC_gt_storage=test_ROSE_DVC_orgGt;
test_rose_DVC_gt_storage(find(test_rose_DVC_gt_storage))=1;

% ROSE SDVC dataset
[fold_train_rose_SDVC_data_storage, fold_valid_rose_SDVC_data_storage, ...
 fold_train_rose_SDVC_gt_storage, fold_valid_rose_SDVC_gt_storage]=data_en_face_ROSE_SDVC_fold_creator(train_ROSE_SDVC_org, train_ROSE_SDVC_orgGt, valid_ROSE_SDVC_org, valid_ROSE_SDVC_orgGt,num_fold);
fold_train_rose_SDVC_gt_storage(find(fold_train_rose_SDVC_gt_storage))=1;
fold_valid_rose_SDVC_gt_storage(find(fold_valid_rose_SDVC_gt_storage))=1;
test_rose_SDVC_data_storage=test_ROSE_SDVC_org;
test_rose_SDVC_gt_storage=test_ROSE_SDVC_orgGt;
test_rose_SDVC_gt_storage(find(test_rose_SDVC_gt_storage))=1;

% ROSE2 dataset
[fold_train_rose2_data_storage, fold_valid_rose2_data_storage, ...
 fold_train_rose2_gt_storage, fold_valid_rose2_gt_storage]=data_en_face_ROSE2_fold_creator(train_ROSE2_org, train_ROSE2_orgGt, valid_ROSE2_org, valid_ROSE2_orgGt,num_fold);
fold_train_rose2_gt_storage(find(fold_train_rose2_gt_storage))=1;
fold_valid_rose2_gt_storage(find(fold_valid_rose2_gt_storage))=1;
test_rose2_data_storage=test_ROSE2_org;
test_rose2_gt_storage=test_ROSE2_orgGt;
test_rose2_gt_storage(find(test_rose2_gt_storage))=1;


%% Phase 1: conduct data augmentation
% training dataset
% ROSE SVC dataset
[augmentated_train_rose_SVC_octa_data, augmentated_train_rose_SVC_octa_gt_data, augmentated_train_rose_SVC_octa_thick_gt_data, augmentated_train_rose_SVC_octa_thin_gt_data]=data_enface_ROSE_SVC_augmentator(fold_train_rose_SVC_data_storage, fold_train_rose_SVC_gt_storage, fold_train_rose_SVC_thick_gt_storage, fold_train_rose_SVC_thin_gt_storage);
% ROSE DVC dataset
[augmentated_train_rose_DVC_octa_data, augmentated_train_rose_DVC_octa_gt_data]=data_enface_ROSE_DVC_augmentator(fold_train_rose_DVC_data_storage, fold_train_rose_DVC_gt_storage);
% ROSE SDVC dataset
[augmentated_train_rose_SDVC_octa_data, augmentated_train_rose_SDVC_octa_gt_data]=data_enface_ROSE_SDVC_augmentator(fold_train_rose_SDVC_data_storage, fold_train_rose_SDVC_gt_storage);
% ROSE2 dataset
% [augmentated_train_rose2_octa_data, augmentated_train_rose2_octa_gt_data]=data_enface_ROSE2_augmentator(fold_train_rose2_data_storage, fold_train_rose2_gt_storage, num_aug);

% validation dataset
% ROSE SVC dataset
% [augmentated_valid_rose_SVC_octa_data, augmentated_valid_rose_SVC_octa_gt_data, augmentated_valid_rose_SVC_octa_thick_gt_data, augmentated_valid_rose_SVC_octa_thin_gt_data]=data_enface_ROSE_SVC_augmentator(fold_valid_rose_SVC_data_storage, fold_valid_rose_SVC_gt_storage, fold_valid_rose_SVC_thick_gt_storage, fold_valid_rose_SVC_thin_gt_storage);
% ROSE DVC dataset
% [augmentated_valid_rose_DVC_octa_data, augmentated_valid_rose_DVC_octa_gt_data]=data_enface_ROSE_DVC_augmentator(fold_valid_rose_DVC_data_storage, fold_valid_rose_DVC_gt_storage);
% ROSE SDVC dataset
% [augmentated_valid_rose_SDVC_octa_data, augmentated_valid_rose_SDVC_octa_gt_data]=data_enface_ROSE_SDVC_augmentator(fold_valid_rose_SDVC_data_storage, fold_valid_rose_SDVC_gt_storage);
% ROSE2 dataset
% [augmentated_valid_rose2_octa_data, augmentated_valid_rose2_octa_gt_data]=data_enface_ROSE2_augmentator(fold_valid_rose2_data_storage, fold_valid_rose2_gt_storage, num_aug);


%% Phase 1: launch the enface segmentation model
% coarse neural network
% SVC segmentation network
SVC_coarse_network=coarse_stage_prototype_4_SVC(augmentated_train_rose_SVC_octa_data);
figure;
plot(SVC_coarse_network)
title("SVC_coarse_network")

% DVC segmentation network
DVC_coarse_network=coarse_stage_prototype_9_DVC(augmentated_train_rose_DVC_octa_data);
figure;
plot(DVC_coarse_network)
title("DVC_coarse_network")

% SDVC segmentation
% SDVC_coarse_network=rose_1_model_SDVC_coarse_launcher(augmentated_train_rose_SDVC_octa_data);
% figure;
% plot(SDVC_coarse_network)
% title("SDVC_coarse_network")

% fine neural network
% SVC segmentation network
SVC_fine_network=fine_stage_prototype_SVC(augmentated_train_rose_SVC_octa_data);
% SVC_fine_network=rose_1_model_SVC_fine_launcher(augmentated_train_rose_SVC_octa_data);
figure;
plot(SVC_fine_network)
title("SVC_fine_network")

% DVC segmentation network
DVC_fine_network=fine_stage_prototype_DVC(augmentated_train_rose_DVC_octa_data);
% DVC_fine_network=rose_1_model_DVC_fine_launcher(augmentated_train_rose_DVC_octa_data);
figure;
plot(DVC_fine_network)
title("DVC_fine_network")

% SDVC segmentation network
% SDVC_fine_network=rose_1_model_SDVC_fine_launcher(augmentated_train_rose_SDVC_octa_data);
% figure;
% plot(SDVC_fine_network)
% title("SDVC_fine_network")


%% Phase 1: train segmentation neural network
%% Coarse stage model training
% check if the trained model is already generated
% SVC coarse model
stored_data_path="~/data/klee232/trained_en_face_seg_model/";
prototype_name="trained_coarse_stage_model_4_Noweightdecay_";
layer_SVC="SVC_";
augemented="triaug";
loss_function="_msePlustversky.mat";
complete_file_name_SVC=strcat(stored_data_path,prototype_name,layer_SVC,augemented,loss_function);

% DVC coarse model
stored_data_path="~/data/klee232/trained_en_face_seg_model/";
prototype_name="trained_coarse_stage_model_9_";
layer_DVC="DVC_";
augemented="triaug";
loss_function="_msePlustversky.mat";
complete_file_name_DVC=strcat(stored_data_path,prototype_name,layer_DVC,augemented,loss_function);

% SVC coarse trained model
if isfile(complete_file_name_SVC)
    load(complete_file_name_SVC);   
    trained_SVC_coarse_network=net_storage;
end
% DVC coarse trained model
if isfile(complete_file_name_DVC)
    load(complete_file_name_DVC);  
    trained_DVC_coarse_network=net_storage;
end

% decide whether you want to train again
% SVC coarse model
train_SVC_ind="y"; % y: for train, others: no train
if train_SVC_ind=="y"
    num_epoch=200;
    learning_rate=0.0005;
    lyr_SVC="SVC";
    % coarse model training
    trained_SVC_coarse_network=training_function_SVC_coarse_f4(SVC_coarse_network, augmentated_train_rose_SVC_octa_data, augmentated_train_rose_SVC_octa_thick_gt_data, augmentated_train_rose_SVC_octa_thin_gt_data, augmentated_valid_rose_SVC_octa_data, augmentated_valid_rose_SVC_octa_thick_gt_data, augmentated_valid_rose_SVC_octa_thin_gt_data,num_epoch,learning_rate,lyr_SVC);
end

%%
% DVC coarse model
train_DVC_ind="n"; % y: for train, others: no train
if train_DVC_ind=="y"
    num_epoch=200;
    learning_rate=0.0005;
    lyr_DVC="DVC";
    % coarse model training
    trained_DVC_coarse_network=training_function_DVC_coarse(DVC_coarse_network, augmentated_train_rose_DVC_octa_data, augmentated_train_rose_DVC_octa_gt_data, augmentated_valid_rose_DVC_octa_data, augmentated_valid_rose_DVC_octa_gt_data,num_epoch,learning_rate,lyr_DVC);
end


%% fine stage model training
% check if the trained model is already generated
% SVC model 
stored_data_path="~/data/klee232/trained_en_face_seg_model/";
fine_prototype_name="trained_fine_prototype_model_256_";
coarse_prototype_name="coarse_model_4_";
layer_SVC="SVC";
augemented="_triaug";
loss_function="_tversky_bce.mat";
complete_file_name_SVC=strcat(stored_data_path,fine_prototype_name, coarse_prototype_name,layer_SVC,augemented,loss_function);

% DVC model
stored_data_path="~/data/klee232/trained_en_face_seg_model/";
fine_prototype_name="trained_fine_prototype_DVC_model_256_";
coarse_prototype_name="coarse_model_9_";
layer_DVC="DVC";
augemented="_triaug";
loss_function="_tversky_bce.mat";
complete_file_name_DVC=strcat(stored_data_path,fine_prototype_name, coarse_prototype_name,layer_DVC,augemented,loss_function);

% if the fine models exist, load the fine models
% SVC fine trained model
if isfile(complete_file_name_SVC)
    load(complete_file_name_SVC); 
    trained_SVC_fine_network=net_storage;
end
% DVC fine trained model
if isfile(complete_file_name_DVC)
    load(complete_file_name_DVC);    
    trained_DVC_fine_network=net_storage;
end

% select the best coarse model
best_trained_SVC_coarse_network=trained_SVC_coarse_network{3,1};
best_trained_DVC_coarse_network=trained_DVC_coarse_network{3,1};

% fine model training
train_SVC_ind="n"; % y: for train, others: no train 
if train_SVC_ind=="y"
    num_epoch=200;
    learning_rate=0.001;
    lyr_SVC='SVC';
    trained_SVC_fine_network=training_function_SVC_fine(SVC_fine_network, best_trained_SVC_coarse_network, augmentated_train_rose_SVC_octa_data, augmentated_train_rose_SVC_octa_gt_data, augmentated_valid_rose_SVC_octa_data, augmentated_valid_rose_SVC_octa_gt_data,num_epoch,learning_rate,lyr_SVC);
end
train_DVC_ind="n"; % y: for train, others: no train
if train_DVC_ind=="y"
    num_epoch=200;
    learning_rate=0.001;
    lyr_DVC='DVC';
    trained_DVC_fine_network=training_function_DVC_fine(DVC_fine_network, best_trained_DVC_coarse_network, augmentated_train_rose_DVC_octa_data, augmentated_train_rose_DVC_octa_gt_data, augmentated_valid_rose_DVC_octa_data, augmentated_valid_rose_DVC_octa_gt_data,num_epoch,learning_rate,lyr_DVC);
end

% select the best fine network
best_trained_SVC_fine_network=trained_SVC_fine_network{3,1};
best_trained_DVC_fine_network=trained_DVC_fine_network{4,1};


%% Phase 1 test segmentation network
% ROC and AUC
testing_function_ROC(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_ROC(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");

% Pixel Prediction Accuracy
testing_function_pixel_accuracy(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_pixel_accuracy(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");

% G-mean
testing_function_pixel_Gmean(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_pixel_Gmean(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");

% Kappa
testing_function_pixel_Kappa(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_pixel_Kappa(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");

% Dice
testing_function_pixel_diceScore(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_pixel_diceScore(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");

% FDR
testing_function_pixel_FDR(best_trained_SVC_coarse_network, best_trained_SVC_fine_network, test_rose_SVC_data_storage, test_rose_SVC_gt_storage, "SVC");
testing_function_pixel_FDR(best_trained_DVC_coarse_network, best_trained_DVC_fine_network, test_rose_DVC_data_storage, test_rose_DVC_gt_storage, "DVC");


%% Phase 2 Input Data Process (more in progress)
addpath(genpath("Phase_Input_Data_Process/"))

% this part serves as an additional process for input image for neural
% network. 
% planned task:
% 1. octa region sorting: a function that is able to sort the octa image
% into retinal quardants

addpath(genpath("Phase_2_Input_Data_Process/"))

% load additional input feature - branch vessel length

% create labels for prediction model
[label_gt_storage]=data_labelCreator();


%% 3D OCTA partitioning
% partition and conver the groundtruth storage variable to array type
% (conduct only necessary)
% if isfile("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi.mat")
%     load("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi.mat");
%     load("~/data/klee232/processed_data/octa gt arrays/valid_octa_gt_data_complete_choroid_excluded_frangi.mat");
%     load("~/data/klee232/processed_data/octa gt arrays/test_octa_gt_data_complete_choroid_excluded_frangi.mat");
% else
%     Num_fold=4;
%     Test_ratio=0.25;
%     [train_octa_gt_storage, valid_octa_gt_storage, test_octa_gt_storage,...
%      ~, ~, ~]=data_partitioner(octa_gt_storage,label_gt_storage,Num_fold,Test_ratio);
%     save("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi.mat","train_octa_gt_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa gt arrays/valid_octa_gt_data_complete_choroid_excluded_frangi.mat","valid_octa_gt_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa gt arrays/test_octa_gt_data_complete_choroid_excluded_frangi.mat","test_octa_gt_storage","-v7.3");
% end

% % partition and convert the branch storage variable to array type
% if isfile("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat")
%     load("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat");
%     load("~/data/klee232/processed_data/octa gt arrays/valid_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat");
%     load("~/data/klee232/processed_data/octa gt arrays/test_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat");
%     load("~/data/klee232/processed_data/octa label arrays/train_label_gt.mat");    
%     load("~/data/klee232/processed_data/octa label arrays/valid_label_gt.mat");
%     load("~/data/klee232/processed_data/octa label arrays/test_label_gt.mat");
% else
%     Num_fold=4;
%     Test_ratio=0.25;
%     [train_octa_branch_length_storage, valid_octa_branch_length_storage, test_octa_branch_length_storage,...
%      train_label_gt_storage, valid_label_gt_storage, test_label_gt_storage]=data_partitioner(octa_branch_length_storage,label_gt_storage,Num_fold,Test_ratio);
%     save("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat","train_octa_branch_length_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa gt arrays/valid_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat","valid_octa_branch_length_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa gt arrays/test_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat","test_octa_branch_length_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa label arrays/train_label_gt.mat","train_label_gt_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa label arrays/valid_label_gt.mat","valid_label_gt_storage","-v7.3");
%     save("~/data/klee232/processed_data/octa label arrays/test_label_gt.mat","test_label_gt_storage","-v7.3");
% end

% % conduct data augmentation on training and validation data
% [rotated_train_octa_branch_length_storage, rotated_train_label_gt_storage]=image_augmentation_rotator(train_octa_branch_length_storage, train_label_gt_storage);
% [rotated_valid_octa_branch_length_storage, rotated_valid_label_gt_storage]=image_augmentation_rotator(valid_octa_branch_length_storage, valid_label_gt_storage);


%% 2D OCTA data partitioning
% create folded dataset

[en_face_vessel_data_storage]=data_en_face_feature_generator(en_face_octa_seg_storage);


Num_fold=4;
Test_ratio=0.25;
[train_en_face_octa_storage, valid_en_face_octa_storage, test_en_face_octa_storage,...
 train_label_gt_storage,valid_label_gt_storage,test_label_gt_storage]=data_en_face_partitioner(en_face_octa_storage,label_gt_storage,Num_fold,Test_ratio);

Num_fold=4;
Test_ratio=0.25;
[train_en_face_data_storage, valid_en_face_data_storage, test_en_face_data_storage,...
 ~,~,~]=data_en_face_partitioner(en_face_octa_seg_storage,label_gt_storage,Num_fold,Test_ratio);


Num_fold=4;
Test_ratio=0.25;
[train_en_face_vessel_data_storage, valid_en_face_vessel_data_storage, test_en_face_vessel_data_storage,...
 ~,~,~]=data_en_face_partitioner(en_face_vessel_data_storage,label_gt_storage,Num_fold,Test_ratio);

num_feature=3;
train_fusion_data_storage=zeros(size(train_en_face_vessel_data_storage,1)*num_feature,size(train_en_face_vessel_data_storage,2),size(train_en_face_vessel_data_storage,3), size(train_en_face_vessel_data_storage,4), size(train_en_face_vessel_data_storage,5));
valid_fusion_data_storage=zeros(size(valid_en_face_vessel_data_storage,1)*num_feature,size(valid_en_face_vessel_data_storage,2),size(valid_en_face_vessel_data_storage,3), size(valid_en_face_vessel_data_storage,4), size(valid_en_face_vessel_data_storage,5));
train_fusion_data_storage(1:num_feature:end,:,:,:,:)=train_en_face_octa_storage;
train_fusion_data_storage(2:num_feature:end,:,:,:,:)=train_en_face_data_storage;
train_fusion_data_storage(3:num_feature:end,:,:,:,:)=train_en_face_vessel_data_storage;
valid_fusion_data_storage(1:num_feature:end,:,:,:,:)=valid_en_face_octa_storage;
valid_fusion_data_storage(2:num_feature:end,:,:,:,:)=valid_en_face_data_storage;
valid_fusion_data_storage(3:num_feature:end,:,:,:,:)=valid_en_face_vessel_data_storage;


%% Phase 3 Disease Prediction neural network training and testing
% this part serves as the neural network training and testing parts for our
% classification neural network
addpath(genpath("Phase_3_Prediction_model/"))

% change variable type
% train_en_face_data_storage=single(train_en_face_data_storage);
% valid_en_face_data_storage=single(valid_en_face_data_storage);
% test_en_face_data_storage=single(test_en_face_data_storage);

% train_en_face_vessel_storage=single(train_en_face_vessel_data_storage);
% valid_en_face_vessel_storage=single(valid_en_face_vessel_data_storage);
% test_en_face_vessel_storage=single(test_en_face_vessel_data_storage);

train_fusion_data_storage=single(train_fusion_data_storage);
valid_fusion_data_storage=single(valid_fusion_data_storage);
test_fusion_vessel_storage=single(test_en_face_vessel_data_storage);



% launch the neural network
classes=unique(train_label_gt_storage);
num_classes=size(classes,1);
% num_feature=(size(train_octa_gt_storage,4)+size(train_octa_branch_length_storage,4))/size(train_label_gt_storage,1);
[img_depth, img_row, img_col,~]=size(train_fusion_data_storage);
network=original_network_mean_removal_pca_net_launcher(img_depth, img_row, img_col, num_classes);
plot(network)

% start the neural network training
num_epoch=100;
learning_rate=0.0005;
trained_network_en_face_imgSeg=training_function(network,train_fusion_data_storage,train_label_gt_storage,...
                                                             valid_fusion_data_storage,valid_label_gt_storage,...
                                                             num_epoch,learning_rate);

load("~/data/klee232/trained_model/trained_network_en_face_img_seg_smaller.mat")

% start the neural network testing
[test_loss,confusion_matrix]=testing_function(trained_original_network_mean_removal_pca,test_octa_gt_storage,test_label_gt_storage,Num_fold);

