

function [augmentated_data, augmentated_gt_data]=data_enface_ROSE_DVC_augmentator(data, gt_data)
    %% conduct data augmentation (image rotation)
    % num_aug=3;
    % [augmentated_data, augmentated_gt_data]=image_rose_data_augmentors_fold_img_rotator(data, gt_data, num_aug);
    

    %% conduct data augmentation (image elastic deformations)
    % alpha=15;
    % sigma=5;
    % [augmentated_data, augmentated_gt_data]=image_rose_data_augmentors_fold_img_elastic_deformer(augmentated_data, augmentated_gt_data, alpha, sigma);


    %% conduct data augmentation (image random contrast jitter)
    % gamma_range=[0.8,1.2];
    % beta_range=[-0.1,0.1];
    % [augmentated_data, augmentated_gt_data]=image_rose_data_augmentors_fold_img_contrast_jitter(augmentated_data, augmentated_gt_data, gamma_range, beta_range);


    %% conduct data augmentation (image flipping)
    [augmentated_data, augmentated_gt_data]=image_rose_data_augmentors_fold_img_flipper(data, gt_data);

end