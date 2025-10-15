% Created by Kuan-Min Lee
% Created date:: May 23rd, 2024

% Brief User Introduction:
% (revised from Morgan collab)
% This module utilize a
% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% picture_obj_octa: struct to store the process information
% filtered_OCTA_image: input 3D image (3D image array)
% filtered_seg_mask: mask for segmentation layer (3D mask array)

% Output:
% picture_obj_octa: struct to store the process information (struct)
% OCTA_img_den: outcome of denoised image (denoised 3D image) 


function  [picture_obj_octa,OCTA_img_den]=image_groundtruth_generator_denoise(picture_obj_octa,filtered_OCTA_img,mask_orig)
    %% partitioning the octa image into surface, and deep capillary layer
    % create corresponding binary mask
    mask_surf=zeros(size(mask_orig));
    mask_deep=zeros(size(mask_orig));
    mask_surf(mask_orig>=2 & mask_orig<=6)=1;
    mask_deep(mask_orig==8)=1;
    % partition the the octa image into three corresponding part
    OCTA_surf=filtered_OCTA_img.*double(mask_surf);
    OCTA_deep=filtered_OCTA_img.*double(mask_deep);

    %% Conduct Edge detection     
    % retrieve the dimensional information
    num_slice=size(OCTA_surf,1);
    num_row=size(OCTA_surf,2);
    num_col=size(OCTA_surf,3);

    % create storage variables
    OCTA_surf_edge_horz=zeros(size(OCTA_surf));
    OCTA_surf_edge_vert_1=zeros(size(OCTA_surf));
    OCTA_surf_edge_vert_2=zeros(size(OCTA_surf));
    OCTA_deep_edge_horz=zeros(size(OCTA_deep));
    OCTA_deep_edge_vert_1=zeros(size(OCTA_deep));
    OCTA_deep_edge_vert_2=zeros(size(OCTA_deep));


    % perform edge detection for each slice of 3d image and compute the
    % distance of the two furthest distance
    % generate horizontal edge
    for i_slice=1:num_slice
        % grab out each slice
        current_slice=OCTA_surf(i_slice,:,:);
        current_slice_deep=OCTA_deep(i_slice,:,:);
        current_slice=squeeze(current_slice);
        current_slice_deep=squeeze(current_slice_deep);

        % compute edge for each slice
        % if the current layer is all zeros, skip the edge computation
        if max(current_slice,[],'all')>0
            current_edge=edge(current_slice,'Canny',[0.2 0.6]); % tuned
            OCTA_surf_edge_horz(i_slice,:,:)=current_edge;
        end
        if max(current_slice_deep,[],'all')>0
            current_edge_deep=edge(current_slice_deep,'canny', [0.32 0.9]); % tuned
            OCTA_deep_edge_horz(i_slice,:,:)=current_edge_deep;
        end

    end

    % generate vertical edge in one direction
    for i_row=1:num_row
        % grab out each row
        current_row=OCTA_surf(:,i_row,:);
        current_row_deep=OCTA_deep(:,i_row,:);
        current_row=squeeze(current_row);
        current_row_deep=squeeze(current_row_deep);

        % compute edge for each row
        % compute only when the current slice is not zero
        if max(current_row,[],'all')
            current_edge=edge(current_row,'Canny',[0.3 0.53]); % tuned
            OCTA_surf_edge_vert_1(:,i_row,:)=current_edge;
        end
        if max(current_row_deep,[],'all')
            current_edge_deep=edge(current_row_deep,'canny',[0.34 0.8]); % tuned
            OCTA_deep_edge_vert_1(:,i_row,:)=current_edge_deep;
        end

    end

    for i_col=1:num_col
        % grab out each row
        current_col=OCTA_surf(:,:,i_col);
        current_col_deep=OCTA_deep(:,:,i_col);
        current_col=squeeze(current_col);
        current_col_deep=squeeze(current_col_deep);

        % compute edge for each column
        if max(current_col,[],'all')
            current_edge=edge(current_col,'Canny',[0.35 0.5]); % tuned
            OCTA_surf_edge_vert_2(:,:,i_col)=current_edge;
        end
        if max(current_col_deep,[],'all')
            current_edge_deep=edge(current_col_deep,'Canny',[0.38 0.8]); % tuned
            OCTA_deep_edge_vert_2(:,:,i_col)=current_edge_deep;
        end

    end

    % conduct image closing
    se = strel('disk',3); % tuned
    OCTA_surf_edge_horz=imclose(OCTA_surf_edge_horz,se);
    OCTA_surf_edge_vert_1=imclose(OCTA_surf_edge_vert_1,se);
    OCTA_surf_edge_vert_2=imclose(OCTA_surf_edge_vert_2,se);
    OCTA_deep_edge_horz=imclose(OCTA_deep_edge_horz,se);
    OCTA_deep_edge_vert_1=imclose(OCTA_deep_edge_vert_1,se);
    OCTA_deep_edge_vert_2=imclose(OCTA_deep_edge_vert_2,se);

    % fuse the edge detection outcomes
    OCTA_surf_edge=sqrt(OCTA_surf_edge_horz.^2+sqrt(OCTA_surf_edge_vert_1.^2+OCTA_surf_edge_vert_2.^2).^2);
    OCTA_surf_edge(OCTA_surf_edge<1.414)=0;
    OCTA_deep_edge=sqrt(OCTA_deep_edge_horz.^2+sqrt(OCTA_deep_edge_vert_1.^2+OCTA_deep_edge_vert_2.^2).^2);
    OCTA_deep_edge(OCTA_deep_edge<1.414)=0;

    %% store the outcome
    picture_obj_octa.OCTA_mask=mask_orig;
    OCTA_img_den=zeros(size(filtered_OCTA_img));
    nonzeros_surf_ind=find(OCTA_surf_edge>0);
    nonzeros_surf=OCTA_surf_edge(nonzeros_surf_ind);
    OCTA_img_den(nonzeros_surf_ind)=nonzeros_surf;
    nonzeros_deep_ind=find(OCTA_deep_edge>0);
    nonzeros_deep=OCTA_deep_edge(nonzeros_deep_ind);
    OCTA_img_den(nonzeros_deep_ind)=nonzeros_deep;
    picture_obj_octa.filtered_img_den_surf_deep=OCTA_img_den;

end