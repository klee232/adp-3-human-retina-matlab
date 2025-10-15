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


function [node_branch_storage]=image_feature_en_face_generator_node(data_branch_storage)

    %% extract branch for each file and grab out the nodes where nodes represent the locations where several vessels are connected together
    num_files=size(data_branch_storage,4);
    num_layers=size(data_branch_storage,1);
    
    % set up connectivity mask
    connect_mask=ones(3,3);

    % set up storage variable
    node_branch_storage=zeros(size(data_branch_storage));
    % loop through each file
    for i_file=1:num_files
        current_octa_branch_img=data_branch_storage(:,:,:,i_file);

        for i_layer=1:num_layers
            current_layer_octa_branch_img=logical(current_octa_branch_img(i_layer,:,:));
            current_layer_octa_branch_img=squeeze(current_layer_octa_branch_img);
            
            % use convolution on each file and get the neighbor pixels number for
            % each pixel location
            current_octa_gt_branch_neighbors=conv2(double(current_layer_octa_branch_img),connect_mask,'same')-1;
    
            % grab out the nodes (nodes neighbor numbers > 2) and store the
            % nodes locations
            current_octa_gt_branch_nodes=current_layer_octa_branch_img & (current_octa_gt_branch_neighbors>2);
            node_branch_storage(i_layer,:,:,i_file)=current_octa_gt_branch_nodes;
        end

    end   

end