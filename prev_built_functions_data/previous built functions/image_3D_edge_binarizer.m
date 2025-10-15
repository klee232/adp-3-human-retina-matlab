% Created by Kuan-Min Lee
% Createed date: May 23rd, 2024

% Brief User Introduction:
% (modified from Morgan collab) 
% The following function is utilized to compute conduct edge detection to
% sharpening the segmentation outcome in x, y, and z and conduct image
% fusion using maximum and image closing for better segmentation outcome

% Input parameter:
% BB0: denoised segmentation for OCTA or OCT image

% Output:
% BB_binar: binarization outcome from the input OCTA or OCT

function [BB_binar]=image_3D_edge_binarizer(BB0)
    disp("Image 3D Edge Binarization running...")
    BB_edge_horz=zeros(size(BB0));
    BB_edge_vert_1=zeros(size(BB0));
    BB_edge_vert_2=zeros(size(BB0));
    num_slice=size(BB0,1);
    num_row=size(BB0,2);
    num_col=size(BB0,3);

    %% Edge detection part
    % perform edge detection for each slice of 3d image and compute the
    % distance of the two furthest distance
    % generate horizontal edge
    for i_slice=1:num_slice
        % compute edges for each slice 
        current_slice=BB0(i_slice,:,:);
        current_slice=squeeze(current_slice);
        current_edge=edge(current_slice,'Canny',[0.7 0.9]);
        BB_edge_horz(i_slice,:,:)=current_edge;
    end   
    % generate vertical edge in one direction
    for i_row=1:num_row
        % compute edges for each row
        current_row=BB0(:,i_row,:);
        current_row=squeeze(current_row);
        current_edge=edge(current_row,'Canny',[0.99 0.995]);
        BB_edge_vert_1(:,i_row,:)=current_edge;
    end
    for i_col=1:num_col
        % compute edges for each row
        current_col=BB0(:,:,i_col);
        current_col=squeeze(current_col);
        current_edge=edge(current_col,'Canny',[0.99 0.995]);
        BB_edge_vert_2(:,:,i_col)=current_edge;
    end
    %% fusing edge part with original image
    BB_edge=BB_edge_horz+BB_edge_vert_1;
    BB_edge=BB_edge+BB_edge_vert_2;
    BB_edge(BB_edge>1)=1;
    BB_fuse=BB0+BB_edge;

    %% conduct binarization with original image
    BB_reshape=reshape(BB_fuse,1,[]);
    BB_reshape_sort=sort(BB_reshape);
    num_elements=size(BB_reshape_sort,2);
    thres_ind=floor(0.966*num_elements);
    thres=BB_reshape_sort(1,thres_ind);
    BB_binar=BB_fuse;
    BB_binar(BB_binar<thres)=0;
    BB_binar(BB_binar>=thres)=1;

    disp("Image 3D Edge Binarization completed")
end