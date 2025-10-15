% Created by Kuan-Min Lee
% Created date: Jan. 28th, 2025
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to calculate the vessel width in
% each branch between two nodes

% Input Parameter
% octa_gt_storage: octa groundtruth storage variable generated from
% previous phase
% octa_branch_storage: octa branch storage variable generated from vessel
% skeletionizer

% Output Parameter
% octa_gt_diameter_width_storage: widths projection for octa groundtruth


function [octa_gt_diameter_width_storage]=image_vessel_diameter_calculator(octa_gt_storage,octa_branch_storage)

    %% compute diameter of each section of vessel
    % create storage variable
    num_files=size(octa_gt_storage,1);
    octa_gt_diameter_width_storage=cell(num_files,1);

    % compute diameter calculation
    for i_file=1:num_files
        % grab out current groundtruth and branch image
        current_octa_gt_img=octa_gt_storage{i_file};
        if iscell(current_octa_gt_img)
            current_octa_gt_img=cell2mat(current_octa_gt_img);
        end
        current_octa_branch_img=octa_branch_storage{i_file};
        if iscell(current_octa_branch_img)
            current_octa_branch_img=cell2mat(current_octa_branch_img);
        end

        % compute euclidean distance transformation of all non-binarized
        % points to circle out the vessel regions
        distance_current_octa_gt_mask=bwdist(~current_octa_gt_img);

        % extract local radius region with skeletion points
        radius_current_octa_gt_mask=distance_current_octa_gt_mask.*current_octa_branch_img;

        % compute diameters for the binarized points
        diameter_current_octa_gt_mask=2*radius_current_octa_gt_mask;

        % store the outcome back to storage variable
        octa_gt_diameter_width_storage{i_file}=diameter_current_octa_gt_mask;

    end


end