% Created by Kuan-Min Lee
% Created date: Jan. 28th, 2025
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to calculate the vessel length in
% each branch between two nodes

% Input Parameter
% octa_gt_storage: octa groundtruth storage variable generated from
% previous phase
% octa_branch_storage: octa branch storage variable generated from vessel
% skeletionizer
% octa_nodes_storage: octa nodes storage variable generated from vessel
% skeletionizer

% Output Parameter
% octa_gt_length_storage: lengths projection for octa groundtruth
% octa_branch_length_storage: lengths projection for octa branch


function [octa_gt_length_storage,octa_branch_length_storage]=image_vessel_length_calculator(octa_gt_storage,octa_branch_storage,octa_nodes_storage)

    %% load one r1 file first to grab out the pixel size in xy direction and z direction
    load('/oscar/data/jlee123/procdata/adp-3-human-retina/A-segment/101-1003-y1/473-p1a.mat');
    pixel_size_x=p1.hdr.ScaleX;
    pixel_size_z=p1.hdr.ScaleZ;


    %% calculate vessel length
    num_files=size(octa_branch_storage,1);
    
    % setup voxel size
    voxel_size=[pixel_size_z, pixel_size_x, pixel_size_x];

    % setup outcome legnth branch storage
    octa_gt_length_storage=octa_gt_storage;
    octa_branch_length_storage=cell(num_files,1);
    for i_file=1:num_files
        % grab out branch and nodes image
        current_octa_gt_img=octa_gt_storage{i_file};
        current_octa_branch_img=octa_branch_storage{i_file};
        current_octa_node_img=octa_nodes_storage{i_file};

        % remove nodes from branch image
        current_octa_branch_img_filtered=current_octa_branch_img & ~ current_octa_node_img;

        % label connected components (branches)
        current_octa_branch_labels=bwconncomp(current_octa_branch_img_filtered, 26); % 26-connectivity for 3D

        current_octa_branch_img=double(current_octa_branch_img);

        % grab out the branch voxels
        current_octa_branch_voxels=current_octa_branch_labels.PixelIdxList;
        current_octa_branch_voxels_num=current_octa_branch_labels.NumObjects;

        % loop through each branch and calculate their lengths
        for i_node=1:current_octa_branch_voxels_num
            current_octa_branch_voxels_inn=current_octa_branch_voxels{i_node};

            % convert linear indices to subscripts
            [branch_z, branch_x, branch_y]=ind2sub(size(current_octa_branch_img),current_octa_branch_voxels_inn);

            % adjust pixel to real-time scale
            branch_z=branch_z*voxel_size(1);
            branch_x=branch_x*voxel_size(2);
            branch_y=branch_y*voxel_size(3);

            % calculate euclidean distance between consecutive points
            current_octa_branch_distance=sqrt(diff(branch_z).^2+diff(branch_x).^2+diff(branch_y).^2);
            current_octa_branch_distance=sum(current_octa_branch_distance);

            % mark it in the outcome image
            current_octa_branch_img(current_octa_branch_voxels_inn)=current_octa_branch_distance;

            % % create distance map to centerline for binarization image
            % example_branch_mask=zeros(size(current_octa_branch_img,1),size(current_octa_branch_img,2),size(current_octa_branch_img,3));
            % example_branch_mask(current_octa_branch_voxels_inn)=1;
            % example_branch_mask=example_branch_mask>0;
            % distance_current_octa_branch_voxel_inn=bwdist(example_branch_mask);
            % distance_current_octa_branch_real_inn=distance_current_octa_branch_voxel_inn.*sqrt(pixel_size_z^2+pixel_size_x^2+pixel_size_x^2);
            % distance_current_octa_branch_real_mask=distance_current_octa_branch_real_inn<=50e-3;
            % 
            % % multiply the mask with current ground truth image and assign
            % % the value
            % current_octa_gt_branch=current_octa_gt_img.*distance_current_octa_branch_real_mask;
            % current_octa_gt_img(find(current_octa_gt_branch))=current_octa_branch_distance;
        end

        % store the final outcome for current image
        octa_gt_length_storage{i_file}=current_octa_gt_img;
        octa_branch_length_storage{i_file}=current_octa_branch_img;

    end

    % save the processed data inside the folder
    save("~/data/klee232/processed_data/pad_octa_gt_data_complete_choroid_excluded_frangi_vessel_length.mat","octa_gt_length_storage",'-v7.3');
    save("~/data/klee232/processed_data/pad_octa_gt_data_complete_choroid_excluded_frangi_vessel_length_branch.mat","octa_branch_length_storage","-v7.3");

end