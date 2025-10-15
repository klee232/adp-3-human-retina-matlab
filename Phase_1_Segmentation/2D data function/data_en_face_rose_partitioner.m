% Created by Kuan-Min Lee
% Created date: Oct. 30th, 2024
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to data partitioning for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% octa_storage: octa storage variable generated from phase 1
% octa_gt_storage: octa groundtruth storage variable generated from phase 1

% Output Parameter
% train_octa_storage: training octa storage variable 
% valid_octa_storage: validation octa storage variable
% test_octa_storage: testing octa storage variable
% train_octa_gt_storage: training octa groundtruth storage variable
% valid_octa_gt_storage: validation octa groundtruth storage variable
% test_octa_gt_storage: testing octa groundtruth storage variable


function [train_data_storage, valid_data_storage, test_data_storage,...
          train_gt_storage,valid_gt_storage,test_gt_storage]=data_en_face_rose_partitioner(data_storage,gt_storage,num_fold,Test_ratio)
    
    %% retireve the number of data in the original dataset
    num_file=size(data_storage,4);

    %% setup the ratio of testing and validation data
    test_ratio=Test_ratio;
    valid_ratio=1/num_fold;
    % default valid ratio as 0.25
    if valid_ratio==1
        valid_ratio=0.25;
    end
    num_test_file=floor(test_ratio*num_file);
    num_remain_file=num_file-num_test_file;
    num_valid_file=floor(valid_ratio*num_remain_file);
    num_train_file=num_remain_file-num_valid_file; 


    %% form the indices for training, validation, and testing data
    % train indices
    train_ind=zeros(num_fold,num_train_file);
    
    %% create storage variables
    example_file=squeeze(data_storage(:,:,:,1));
    [img_depth,img_row,img_col]=size(example_file);
    % data storage
    train_data_storage=zeros(img_depth, img_row, img_col, num_train_file,num_fold);
    valid_data_storage=zeros(img_depth, img_row, img_col, num_valid_file,num_fold);
    test_data_storage=zeros(img_depth, img_row, img_col, num_test_file,1);

    % gt storage
    train_gt_storage=zeros(img_depth, img_row, img_col, num_train_file,num_fold);
    valid_gt_storage=zeros(img_depth, img_row, img_col, num_valid_file,num_fold);
    test_gt_storage=zeros(img_depth, img_row, img_col, num_test_file,1);


    %% form testing dataset
    test_ind=(num_file-num_test_file+1):num_file;
    test_data_storage(:,:,:,:,1)=data_storage(:,:,:,test_ind,1);
    test_gt_storage(:,:,:,:,1)=gt_storage(:,:,:,test_ind,1);


    %% form folded dataset (training and validation dataset)
    % filter out testing index
    overall_ind=1:num_file;
    train_overall_ind=setdiff(overall_ind,test_ind);

    % create starter index fold dataset
    valid_offset=num_valid_file;
    for i_fold=1:num_fold
        % form validation dataset
        valid_ind=train_overall_ind((i_fold-1)*valid_offset+1.... 
                                    :(i_fold-1)*valid_offset+num_valid_file);
        valid_data_storage(:,:,:,:,i_fold)=data_storage(:,:,:,valid_ind);  
        valid_gt_storage(:,:,:,:,i_fold)=gt_storage(:,:,:,valid_ind);

        % form training dataset
        train_ind=setdiff(train_overall_ind,valid_ind);
        train_data_storage(:,:,:,:,i_fold)=data_storage(:,:,:,train_ind); 
        train_gt_storage(:,:,:,:,i_fold)=gt_storage(:,:,:,train_ind);

    end
    

end