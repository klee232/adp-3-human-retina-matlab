

function [augmentated_data, augmentated_gt_data]=data_enface_rose_augmentator(data, gt_data, num_aug)
    %% retrieve dimensional information
    num_row=size(data,1);
    num_col=size(data,2);
    num_sample=size(data,3);
    num_fold=size(data,4);
    
    augmentated_data=zeros(num_row,num_col, num_sample*num_aug, num_fold);
    augmentated_gt_data=zeros(num_row,num_col, num_sample*num_aug, num_fold);

    
    %% conduct data augmentation 
    for i_fold=1:num_fold
        current_sample_ind=1;
        for i_sample=1:num_sample
            current_data=data(:,:,i_sample,i_fold);
            current_gt_data=gt_data(:,:,i_sample,i_fold);
            current_data=squeeze(current_data);
            current_gt_data=squeeze(current_gt_data);
            for i_aug=1:num_aug
                aug_ang=10*randn;
                current_aug_data=imrotate(current_data,aug_ang,'bilinear','crop');
                current_aug_gt_data=imrotate(current_gt_data,aug_ang,'bilinear','crop');
                augmentated_data(:,:,current_sample_ind,i_fold)=current_aug_data;
                augmentated_gt_data(:,:,current_sample_ind,i_fold)=current_aug_gt_data;
                current_sample_ind=current_sample_ind+1;
            end
        end
    end




end