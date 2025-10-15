% Created by Kuan-Min Lee
% Created date: Jan. 21st, 2024
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to generate synthsized data based on
% HR. This function will take HR OCTA groundtruth and conduct image
% dilation to generate synthesized MCI OCTA image

% Input Parameter
% octa_gt_storage: octa groundtruth generated from previous phase
% label_gt_storage: label for each sample

% Output Parameter
% synthetic_octa_gt_storage: synthesized octa groundtruth dataset that
% contains both original and synthesized groundtruth data
% synthetic_label_gt_storage: synthesized label dataset that contains both
% original and synthesized label data


function [synthetic_octa_storage,synthetic_octa_gt_storage,synthetic_label_gt_storage]=data_synthetic_generator_HR_MCI(octa_storage,octa_gt_storage,label_gt_storage)

    %% find the labels of each sample and grab out HR and MCI subjects
    num_elements=size(label_gt_storage,1);

    % create storage variable for HR and MCI
    synthetic_octa_storage={};
    synthetic_octa_gt_storage={};

    % create storage variable for synthetic data
    synthetic_label_gt_storage={};
    synthetic_label_gt_storage=categorical(synthetic_label_gt_storage);

    % loop through each sample and grab out HR and MCI samples
    for i_element=1:num_elements
        current_label_gt_storage=label_gt_storage(i_element);

        % if the current sample is a HR sample, store it into HR_octa_gt_storage 
        if isequal(current_label_gt_storage,'HR')
            synthetic_octa_storage=[synthetic_octa_storage;octa_storage{i_element}];
            synthetic_octa_gt_storage=[synthetic_octa_gt_storage;octa_gt_storage{i_element}];
            synthetic_label_gt_storage=[synthetic_label_gt_storage;current_label_gt_storage];
        end

    end


    %% create syntheic dataset based on HR dataset
    num_HR_elements=size(synthetic_octa_gt_storage,1);

    % create 3d disk for dilation
    radius_dil=3;
    se=strel('sphere',radius_dil);

    % grab out each HR file and create its corresponding synthetic MCI
    % dataset
    for i_HR_elements=1:num_HR_elements
        % grab out the files
        % grab out current HR file
        current_HR_octa_storage=synthetic_octa_storage{i_HR_elements};
        current_HR_octa_gt_storage=synthetic_octa_gt_storage{i_HR_elements};

        % apply dilation
        dilated_current_HR_octa_storage=imdilate(current_HR_octa_storage, se);
        dilated_current_HR_octa_gt_storage=imdilate(current_HR_octa_gt_storage, se);

        % store the current dilated image
        synthetic_octa_storage=[synthetic_octa_storage;dilated_current_HR_octa_storage];
        synthetic_octa_gt_storage=[synthetic_octa_gt_storage;dilated_current_HR_octa_gt_storage];

        % concatenate one 'MCI' categorized element into synthetic label
        % storage variable
        synthetic_label_gt_storage=[synthetic_label_gt_storage;'MCI'];
    end


end