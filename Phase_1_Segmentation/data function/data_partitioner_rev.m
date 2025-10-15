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
          train_label_gt_storage,valid_label_gt_storage,test_label_gt_storage]=data_partitioner_rev(data_storage,label_gt_storage,num_feature,num_fold,Test_ratio)
    
    %% retireve the number of data in the original dataset
    num_file=size(data_storage,1);

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

    %% partition the data into corresppnding data (uniformly)
    % grab out the indices for HR, LR, and MCI 
    HR_ind_orig=find(label_gt_storage=="HR");
    LR_ind_orig=find(label_gt_storage=="LR");
    MCI_ind_orig=find(label_gt_storage=="MCI");

    % calculate the intended ratio
    num_HR=size(HR_ind_orig,1);
    num_LR=size(LR_ind_orig,1);
    num_MCI=size(MCI_ind_orig,1);
    ratio_HR=num_HR/(num_HR+num_LR+num_MCI);
    ratio_LR=num_LR/(num_HR+num_LR+num_MCI);

    %% form the indices for training, validation, and testing dataset
    % test indices
    HR_test_num=ceil(num_test_file*ratio_HR);
    LR_test_num=floor(num_test_file*ratio_LR);
    MCI_test_num=num_test_file-HR_test_num-LR_test_num;
    HR_test_ind=HR_ind_orig(1:(1+HR_test_num-1));
    LR_test_ind=LR_ind_orig(1:(1+LR_test_num-1));
    MCI_test_ind=MCI_ind_orig(1:(1+MCI_test_num-1));
    test_ind=[HR_test_ind; LR_test_ind; MCI_test_ind];

    % valid indices
    HR_valid_num=floor(num_valid_file*ratio_HR);
    LR_valid_num=ceil(num_valid_file*ratio_LR);
    MCI_valid_num=num_valid_file-HR_valid_num-LR_valid_num;
    valid_ind=zeros(num_fold,num_valid_file);

    % train indices
    train_ind=zeros(num_fold,num_train_file);
    
    %% create storage variables
    example_file=data_storage{1,1};
    if iscell(example_file)
        example_file=cell2mat(example_file);
    end
    [img_depth,img_row,img_col]=size(example_file);
    % data storage
    train_data_storage=zeros(img_depth, img_row, img_col, num_feature*num_train_file,num_fold);
    valid_data_storage=zeros(img_depth, img_row, img_col, num_feature*num_valid_file,num_fold);
    test_data_storage=zeros(img_depth, img_row, img_col, num_feature*num_test_file,1);

    % label storage
    train_label_gt_storage=categorical(repmat("unknown",num_train_file,num_fold));
    valid_label_gt_storage=categorical(repmat("unknown",num_valid_file,num_fold));
    test_label_gt_storage=categorical(repmat("unknown",num_test_file,1));


    %% form testing dataset
    test_data_storage_cell=data_storage(test_ind,1);
    for i_test=1:num_test_file
        current_test_data_img=test_data_storage_cell{i_test};
        if iscell(current_test_data_img)
            current_test_data_img=cell2mat(current_test_data_img);
            inn_counter=1;
            for i_store=num_feature*(i_test-1)+1:num_feature*(i_test-1)+num_feature
                test_data_storage(:,:,:,i_store,1)=current_test_data_img(inn_counter);
                inn_counter=inn_counter+1;
            end
        else
            inn_counter=1;
            for i_store=num_feature*(i_test-1)+1:num_feature*(i_test-1)+num_feature
                test_data_storage(:,:,:,i_store,1)=current_test_data_img(inn_counter);
                inn_counter=inn_counter+1;
            end
        end
    end
    test_label_gt_storage(:,1)=label_gt_storage(test_ind,1);


    %% form folded dataset (training and validation dataset)
    % filter out testing index
    HR_train_overall_ind=setdiff(HR_ind_orig,HR_test_ind);
    LR_train_overall_ind=setdiff(LR_ind_orig,LR_test_ind);
    MCI_train_overall_ind=setdiff(MCI_ind_orig,MCI_test_ind);

    % create starter index fold dataset
    HR_valid_offset=HR_valid_num;
    LR_valid_offset=LR_valid_num;
    MCI_valid_offset=MCI_valid_num;
    for i_fold=1:num_fold
        % form validation dataset
        HR_valid_ind=HR_train_overall_ind((i_fold-1)*HR_valid_offset+1.... 
                                         :(i_fold-1)*HR_valid_offset+HR_valid_num);
        LR_valid_ind=LR_train_overall_ind((i_fold-1)*LR_valid_offset+1....
                                         :(i_fold-1)*LR_valid_offset+LR_valid_num);
        if i_fold==num_fold
            MCI_valid_ind=MCI_train_overall_ind(size(MCI_train_overall_ind,1)-3....
                                               :end);
        else
            MCI_valid_ind=MCI_train_overall_ind((i_fold-1)*MCI_valid_offset+1....
                                               :(i_fold-1)*MCI_valid_offset+MCI_valid_num);
        end
        valid_ind(i_fold,:)=[HR_valid_ind; LR_valid_ind; MCI_valid_ind];
        valid_data_storage_cell=data_storage(squeeze(valid_ind(i_fold,:)),1);  
        for i_valid=1:num_valid_file
            current_valid_data_img=valid_data_storage_cell{i_valid};
            if iscell(current_valid_data_img)
                inn_counter=1;
                current_valid_data_img=cell2mat(current_valid_data_img);
                for i_store=num_feature*(i_valid-1)+1:num_feature*(i_valid-1)+num_feature
                    valid_data_storage(:,:,:,i_store,i_fold)=current_valid_data_img(inn_counter);
                    inn_counter=inn_counter+1;
                end
            else
                inn_counter=1;
                for i_store=num_feature*(i_valid-1)+1:num_feature*(i_valid-1)+num_feature
                    valid_data_storage(:,:,:,i_store,i_fold)=current_valid_data_img(inn_counter);
                    inn_counter=inn_counter+1;
                end
            end
        end
        valid_label_gt_storage(:,i_fold)=label_gt_storage(valid_ind(i_fold,:),1);

        % form training dataset
        HR_train_ind=setdiff(HR_train_overall_ind,HR_valid_ind);
        LR_train_ind=setdiff(LR_train_overall_ind,LR_valid_ind);
        MCI_train_ind=setdiff(MCI_train_overall_ind,MCI_valid_ind);
        train_ind(i_fold,:)=[HR_train_ind; LR_train_ind; MCI_train_ind];
        train_data_storage_cell=data_storage(squeeze(train_ind(i_fold,:)),1); 
        for i_train=1:num_train_file
            current_train_data_img=train_data_storage_cell{i_train};
            if iscell(current_train_data_img)
                inn_counter=1;
                current_train_data_img=cell2mat(current_train_data_img);
                for i_store=num_feature*(i_train-1)+1:num_feature*(i_train-1)+num_feature
                    train_data_storage(:,:,:,i_store,i_fold)=current_train_data_img(inn_counter);
                    inn_counter=inn_counter+1;
                end
            else
                inn_counter=1;
                for i_store=num_feature*(i_train-1)+1:num_feature*(i_train-1)+num_feature
                    train_data_storage(:,:,:,i_store,i_fold)=current_train_data_img(inn_counter);
                    inn_counter=inn_counter+1;
                end
            end
        end
        train_label_gt_storage(:,i_fold)=label_gt_storage(train_ind(i_fold,:),1);

    end
    

end