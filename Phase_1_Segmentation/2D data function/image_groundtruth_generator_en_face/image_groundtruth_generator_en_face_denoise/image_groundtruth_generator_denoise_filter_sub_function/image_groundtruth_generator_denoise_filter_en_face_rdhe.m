% Created by Kuan-Min Lee
% Created date: Feb. 3rd, 2025
% All rights reserved to Leelab.ai


% Brief User Introduction:
% This function is built to create en face images from 3D images


% Input Parameter:
% data_storage: storage variable (octa_surface, octa_deep, or octa_choroid)
% region_size: size of region for histogram equalization (2-element
% vector)

% Output Parameter:
% denoised_data_storage: storage for denoised images


function  [denoised_data_storage]=image_groundtruth_generator_denoise_filter_en_face_rdhe(data_storage, region_size)          
    
    %% grab out the number of files and create storage variable 
    num_files=size(data_storage,1);

    % grab out dimensional information
    row_size=size(data_storage,2);
    col_size=size(data_storage,3);

    % create storage variable
    denoised_data_storage=zeros(num_files,row_size,col_size);


    %% conduct dynamic regional equalization
    % Determine the number of regions
    region_grid_row=region_size(1);
    region_grid_col=region_size(2);

    % loop through each file and conduct region equalization 
    for i_file=1:num_files
        current_data_img=data_storage(i_file,:,:);
        current_data_img=squeeze(current_data_img);
        for i_row = 1:region_grid_row:row_size
            for i_col = 1:region_grid_col:col_size
                % Define the region boundaries
                region_row_end = min(i_row+region_grid_row-1, row_size);
                region_col_end = min(i_col+region_grid_col-1, col_size);
                
                % Extract local region
                current_data_local_region=current_data_img(i_row:region_row_end, i_col:region_col_end);
                
                % Apply histogram equalization
                current_data_equalized_region=histeq(current_data_local_region);
                
                % Assign back to output image
                denoised_data_storage(i_file,i_row:region_row_end,i_col:region_col_end)=current_data_equalized_region;

            end

        end

    end
    

end