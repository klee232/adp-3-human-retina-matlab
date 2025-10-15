% Created by Kuan-Min Lee
% Created date: Jan. 30th, 2025
% All rights reserved to Leelab.ai

% Brief User Introduction:


% Input Parameter:


% Output Parameter:


function [train_fusion_data_storage, valid_fusion_data_storage, test_fusion_data_storage]=data_fusion(train_data_storage,valid_data_storage, test_data_storage,...
                                                                                                      train_add_data_storage, valid_add_data_storage, test_add_data_storage,...
                                                                                                      train_label_storage, valid_label_storage, test_label_storage)

    %% retrieve dimensional and feature information
    num_train_files=size(train_label_storage,1);
    num_valid_files=size(valid_label_storage,1);
    num_test_files=size(test_label_storage,1);
    num_feature=size(train_data_storage,4)/num_train_files;
    num_feature_add=size(train_add_data_storage,4)/num_train_files;


    %% create storage variable
    train_fusion_data_storage=zeros(size(train_data_storage,1), size(train_data_storage,2), size(train_data_storage,3),...
                                    (size(train_data_storage,4)+size(train_add_data_storage,4)),size(train_data_storage,5));
    valid_fusion_data_storage=zeros(size(valid_data_storage,1), size(valid_data_storage,2), size(valid_data_storage,3),...
                                    (size(valid_data_storage,4)+size(valid_add_data_storage,4)),size(valid_data_storage,5));
    test_fusion_data_storage=zeros(size(test_data_storage,1), size(test_data_storage,2), size(test_data_storage,3),...
                                    (size(test_data_storage,4)+size(test_add_data_storage,4)),size(test_data_storage,5));


    %% fuse the features
    % loop through each file pair and concatenate them in 4th dimension
    % training fusion files
    for i_file=1:num_train_files
        % grab out each file
        current_train_img=train_data_storage(:,:,:,(i_file-1)*num_feature+1:(i_file)*num_feature,:);
        current_train_add_img=train_add_data_storage(:,:,:,(i_file-1)*num_feature_add+1:(i_file)*num_feature_add,:);
        
        % concatenate them in 4th dimension
        current_train_data=cat(4,current_train_img, current_train_add_img);

        % store back to storage
        train_fusion_data_storage(:,:,:,(i_file-1)*(num_feature+num_feature_add)+1:(i_file)*(num_feature+num_feature_add),:)=current_train_data;
        
    end

    % validation fusion files
    for i_file=1:num_valid_files
        % grab out each file
        current_valid_img=valid_data_storage(:,:,:,(i_file-1)*num_feature+1:(i_file)*num_feature,:);
        current_valid_add_img=valid_add_data_storage(:,:,:,(i_file-1)*num_feature_add+1:(i_file)*num_feature_add,:);

        % concatenate them in 4th dimension
        current_valid_data=cat(4,current_valid_img, current_valid_add_img);

        % store back to storage
        valid_fusion_data_storage(:,:,:,(i_file-1)*(num_feature+num_feature_add)+1:(i_file)*(num_feature+num_feature_add),:)=current_valid_data;

    end

    % testing fusion files
    for i_file=1:num_test_files
        % grab out each file
        current_test_img=test_data_storage(:,:,:,(i_file-1)*num_feature+1:(i_file)*num_feature,:);
        current_test_add_img=test_add_data_storage(:,:,:,(i_file-1)*num_feature_add+1:(i_file)*num_feature_add,:);

        % concatenate them in 4th dimension
        current_test_data=cat(4,current_test_img, current_test_add_img);

        % store back to storage
        test_fusion_data_storage(:,:,:,(i_file-1)*(num_feature+num_feature_add)+1:(i_file)*(num_feature+num_feature_add),:)=current_test_data;
    end

    % save the files
    save("~/data/klee232/processed_data/octa gt arrays/train_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch_fused.mat","train_fusion_data_storage","-v7.3");
    save("~/data/klee232/processed_data/octa gt arrays/valid_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch_fused.mat","valid_fusion_data_storage","-v7.3");
    save("~/data/klee232/processed_data/octa gt arrays/test_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch_fused.mat","test_fusion_data_storage","-v7.3");


end
