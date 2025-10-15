


function [vessel_length_en_face_vessel_data_storage, vessel_length_en_face_tortuosity_data_storage]=image_feature_en_face_generator_vessel_length(skeleton_en_face_data_storage, node_en_face_data_storage)

    %% setup pixel size
    load('/oscar/data/jlee123/procdata/adp-3-human-retina/A-segment/101-1003-y1/473-p1a.mat');
    pixel_size_x=p1.hdr.ScaleX;

    % setup voxel size
    voxel_size=[pixel_size_x, pixel_size_x];

    %% compute the vessel length for each image
    num_file=size(skeleton_en_face_data_storage,4);
    num_layer=size(skeleton_en_face_data_storage,1);

    % crete storage variable
    vessel_length_en_face_vessel_data_storage=zeros(size(skeleton_en_face_data_storage));
    vessel_length_en_face_tortuosity_data_storage=zeros(size(skeleton_en_face_data_storage));

    % loop through each file
    for i_file=1:num_file
        current_skeleton_en_face_img=squeeze(skeleton_en_face_data_storage(:,:,:,i_file));
        current_node_en_face_img=squeeze(node_en_face_data_storage(:,:,:,i_file));

        % loop through each layer
        for i_layer=1:num_layer
            current_layer_skeleton_en_face_img=squeeze(current_skeleton_en_face_img(i_layer,:,:));
            current_layer_node_en_face_img=squeeze(current_node_en_face_img(i_layer,:,:));
    
            % label connected components
            current_layer_pruned_skeleton=current_layer_skeleton_en_face_img & ~current_layer_node_en_face_img;
            current_layer_branch_labels=bwconncomp(current_layer_pruned_skeleton, 8); % 26-connectivity for 3D

            current_layer_skeleton_en_face_img=double(current_layer_skeleton_en_face_img);
            current_layer_tortuosity_en_face_img=zeros(size(current_layer_skeleton_en_face_img));

            % grab out the branch voxels
            current_layer_branch_voxels=current_layer_branch_labels.PixelIdxList;
            current_layer_branch_voxels_num=current_layer_branch_labels.NumObjects;


            % loop through each branch and calculate their lengths
            for i_node=1:current_layer_branch_voxels_num
                current_layer_branch_voxels_inn=current_layer_branch_voxels{i_node};

                % convert linear indices to subscripts
                [branch_x, branch_y]=ind2sub(size(current_layer_skeleton_en_face_img),current_layer_branch_voxels_inn);

                % adjust pixel to real-time scale
                branch_x=branch_x*voxel_size(1);
                branch_y=branch_y*voxel_size(2);

                % calculate euclidean distance between consecutive points
                current_octa_branch_distance=sqrt(diff(branch_x).^2+diff(branch_y).^2);
                current_octa_branch_distance=sum(current_octa_branch_distance);

                % compute Euclidean distance between endpoints
                if length(branch_x)>1
                    euclidean_dist=sqrt((branch_x(end)-branch_x(1))^2+(branch_y(end)-branch_y(1))^2);
                else
                    euclidean_dist=1;
                end

                % compute vessel tortuosity
                vessel_tortuosity=current_octa_branch_distance/euclidean_dist;

                % mark it in the outcome image
                current_layer_skeleton_en_face_img(current_layer_branch_voxels_inn)=current_octa_branch_distance;
                current_layer_tortuosity_en_face_img(current_layer_branch_voxels_inn)=vessel_tortuosity;
            end

            vessel_length_en_face_vessel_data_storage(i_layer,:,:,i_file)=current_layer_skeleton_en_face_img;
            vessel_length_en_face_tortuosity_data_storage(i_layer,:,:,i_file)=current_layer_tortuosity_en_face_img;

        end

    end



end