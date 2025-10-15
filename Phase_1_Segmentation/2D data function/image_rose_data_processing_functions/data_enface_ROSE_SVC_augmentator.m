

function [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=data_enface_ROSE_SVC_augmentator(data, gt_data, thick_gt_data, thin_gt_data)
    %% conduct data augmentation (image rotation)
    % num_aug=3;
    % [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_rotator(data, gt_data, thick_gt_data, thin_gt_data, num_aug);
    % 

    %% conduct data augmentation (image elastic deformations)
    % alpha=15;
    % sigma=5;
    % [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_elastic_deformer(augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data, alpha, sigma);
    % 

    %% conduct data augmentation (image random contrast jitter)
    % gamma_range=[0.8,1.2];
    % beta_range=[-0.1,0.1];
    % [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_contrast_jitter(augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data, gamma_range, beta_range);

    %% conduct data augmentation (image flipping)
    % [augmentated_data, augmentated_gt_data, augmentated_thick_gt_data, augmentated_thin_gt_data]=image_rose_data_augmentors_fold_svc_img_flipper(data, gt_data, thick_gt_data, thin_gt_data);


    %% just used original data for augmentated data (for testing only)
    augmentated_data=data;
    augmentated_gt_data=gt_data;
    augmentated_thick_gt_data=thick_gt_data;
    augmentated_thin_gt_data=thin_gt_data;
end