% Created by Kuan-Min Lee
% Created date: Jan. 27th, 2025
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to generate the branch for
% segmentation of octa

% Input Parameter
% octa_gt_storage: octa groundtruth storage variable generated from
% previous

% Output Parameter
% octa_branch_storage: branch for octa groundtruth
% octa_nodes_storage: nodes for octa groundtruth


function [octa_branch_storage,octa_nodes_storage]=image_vessel_skeletionize(octa_gt_storage)

    %% extract branch for each file and grab out the nodes where nodes represent the locations where several vessels are connected together
    num_files=size(octa_gt_storage,1);
    
    % set up connectivity mask
    connect_mask=ones(3,3,3);

    % set up storage variable
    octa_branch_storage=octa_gt_storage;
    octa_nodes_storage=cell(num_files,1);
    % loop through each file
    for i_file=1:num_files
        current_octa_gt_img=octa_gt_storage{i_file};
        current_octa_gt_img=logical(current_octa_gt_img);

        % conduct and store the branch skeletionization and store the
        % outcome
        current_octa_gt_branch=Skeleton3D(current_octa_gt_img);
        octa_branch_storage{i_file}=current_octa_gt_branch;

        % use convolution on each file and get the neighbor pixels number for
        % each pixel location
        current_octa_gt_branch_neighbors=convn(double(current_octa_gt_branch),connect_mask,'same')-1;

        % grab out the nodes (nodes neighbor numbers > 2) and store the
        % nodes locations
        current_octa_gt_branch_nodes=current_octa_gt_branch & (current_octa_gt_branch_neighbors>2);
        octa_nodes_storage{i_file}=current_octa_gt_branch_nodes;

    end   

    % save the processed data inside the folder
    save("~/data/klee232/processed_data/pad_octa_gt_data_complete_choroid_excluded_frangi_vessel_branch.mat","octa_branch_storage","-v7.3");
    save("~/data/klee232/processed_data/pad_octa_gt_data_complete_choroid_excluded_frangi_vessel_node.mat","octa_nodes_storage","-v7.3");

end