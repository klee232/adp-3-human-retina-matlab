% Created by Kuan-Min Lee
% Created date: Jan. 22nd, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is constructed to filtered out all the layers for oct
% images except for surface, deep, and choroid layers

% Input Parameter:
% oct_storage: oct image created from previous phase
% oct_seg_storage: oct segmentation image created from previous phase

% Output Parameter:
% filtered_oct_storage: filtered oct image created from previous phase
% filtered_oct_seg_storage: filtered oct segmentation image created from previous phase


function [filtered_oct_storage,filtered_oct_seg_storage]=data_filter_oct(oct_storage,oct_seg_storage)

    %% create mask for keeping only surface, deep, and choroid layer
    example_oct_img=oct_storage{1};
    example_oct_seg_img=oct_seg_storage{1};
    mask_surf=zeros(size(example_oct_img));
    mask_deep=zeros(size(example_oct_img));
    mask_choroid=zeros(size(example_oct_img));
    mask_surf(example_oct_seg_img>=2 & example_oct_seg_img<=6)=1;
    mask_deep(example_oct_seg_img==8)=1;
    mask_choroid(example_oct_seg_img==14)=1;


    %% keep only surface, deep and choroid layer for oct and oct segmentation images
    num_file=size(oct_storage,1);
    filtered_oct_storage=cell(num_file,1);
    filtered_oct_seg_storage=cell(num_file,1);

    % conduct filtering for each oct image
    for i_file=1:num_file
        % grab out the current oct and oct segmentation file
        current_oct_img=oct_storage{i_file,1};
        current_oct_seg_img=oct_seg_storage{i_file,1};

        % conduct filtering for oct and oct segmentation file
        current_oct_img_surf=current_oct_img.*mask_surf;
        current_oct_img_deep=current_oct_img.*mask_deep;
        current_oct_img_choroid=current_oct_img.*mask_choroid;
        current_oct_seg_img_surf=double(current_oct_seg_img).*mask_surf;
        current_oct_seg_img_deep=double(current_oct_seg_img).*mask_deep;
        current_oct_seg_img_choroid=double(current_oct_seg_img).*mask_choroid;

        % fuse the image
        current_oct_img_filtered=current_oct_img_surf+current_oct_img_deep+current_oct_img_choroid;
        current_oct_seg_img_filtered=current_oct_seg_img_surf+current_oct_seg_img_deep+current_oct_seg_img_choroid;

        % store the filtered image
        filtered_oct_storage{i_file,1}=current_oct_img_filtered;
        filtered_oct_seg_storage{i_file,1}=current_oct_seg_img_filtered;

    end



end