% Created by Kuan-Min Lee
% Createed date: Oct. 22nd, 2024

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]
% 4D image array format: [num_file, num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% octa_storage: processed image storage for OCTA image (4D image array)
% octa_gt_storage: processed image storage for OCTA groundtruth image (4D image array)

% Output:
% octa_gt_storage: padded processed octa groundtruth storage for all files (4D image
% array)
% octa_storage: padded processed octa image storage for all files (4D image
% array)


function [octa_storage,octa_gt_storage]=data_unifier(octa_storage,octa_gt_storage)
    %% retrieve out the largest depth size for each 
    num_files=size(octa_storage,1);
    largest_depth=0;
    for i_file=1:num_files
        % current_file=cell2mat(octa_storage{i_file,1});
        current_file=octa_storage{i_file,1};
        if class(current_file)=='cell'
            current_file=cell2mat(current_file);
        end
        current_file_depth=size(current_file,1);
        % check if the current depth is larger than existed largest depth
        if current_file_depth>largest_depth
            largest_depth=current_file_depth;
        end
    end


    %% padding the smaller image files
    for i_file=1:num_files
        current_file=octa_storage{i_file,1};
        current_gt_file=octa_gt_storage{i_file,1};
        if class(current_file)=='cell'
            current_file=cell2mat(octa_storage{i_file,1});
            current_gt_file=cell2mat(octa_gt_storage{i_file,1});
        end
       
        current_file_depth=size(current_file,1);

        % if the current image file smaller than the largest depth pad the
        % image, pad the image along z direction and store the padded
        % outcome
        if current_file_depth<largest_depth
            pad_current_file=zeros(largest_depth,size(current_file,2),size(current_file,3));
            pad_current_file(1:current_file_depth,:,:)=current_file;
            pad_current_file(current_file_depth+1:end,:,:)=0;
            octa_storage{i_file,1}=pad_current_file;

            pad_current_gt_file=zeros(largest_depth,size(current_gt_file,2),size(current_gt_file,3));
            pad_current_gt_file(1:current_file_depth,:,:)=current_gt_file;
            pad_current_gt_file(current_file_depth+1:end,:,:)=0;
            octa_gt_storage{i_file,1}=pad_current_gt_file;
        else
            pad_current_file=current_file;
            pad_current_gt_file=current_gt_file;
            octa_storage{i_file,1}=pad_current_file;
            octa_gt_storage{i_file,1}=pad_current_gt_file;
        end
    end

    % save the processed data inside the folder
    save("~/data/klee232/processed_data/octa arrays/pad_octa_data_complete_choroid_excluded_frangi.mat","octa_storage",'-v7.3');
    save("~/data/klee232/processed_data/octa gt arrays/pad_octa_gt_data_complete_choroid_excluded_frangi.mat","octa_gt_storage","-v7.3");

end