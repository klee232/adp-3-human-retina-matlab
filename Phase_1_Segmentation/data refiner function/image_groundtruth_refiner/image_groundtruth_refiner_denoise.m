% Created by Kuan-Min Lee
% Createed date: Jan. 15th, 2025

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation part

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]
% 4D image array format: [num_file, num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% orig_octa_gt_storage: groundtruth variable generated from the previous
% phase

% Output:
% den_octa_gt_storage: denoised groundtruth outcome


function [den_octa_gt_storage]=image_groundtruth_refiner_denoise(orig_octa_gt_storage)

    %% check if there existed any temp file

    % if exist temporary files, load it first
    if isfile("Phase_1_Segmentation/data refiner function/temp/den_octa_data_gt.mat") && ...
       isfile("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_temp.mat") && ...
       isfile("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_slice.mat") && ...
       isfile("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_file.mat")
         load("Phase_1_Segmentation/data refiner function/temp/den_octa_data_gt.mat");
         load("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_temp.mat");
         load("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_slice.mat");
         load("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_file.mat");
         
         % start file from last time
         start_file=i_file; 

         % start slice from last time
         start_slice=i_slice; 

    % if not initialize all parameters
    else
        % create storage variable
        den_octa_gt_storage=orig_octa_gt_storage;

        % create temporary storage variable
        example_octa_gt_img=orig_octa_gt_storage{1};

        % start file from last time
        start_file=1;

        % start slice from last time
        start_slice=1;
    end


    %% retireve dimensional information
    [num_file]=size(orig_octa_gt_storage,1);
    [num_slice]=size(example_octa_gt_img,1);


    %% conducting manual denoising for segmentation

    % loop through every file
    for i_file=start_file:num_file
        current_octa_gt=orig_octa_gt_storage{i_file};

        % loop through every slice
        for i_slice=start_slice:num_slice
            %% display message
            mesg=strcat("Current file number: ",string(i_file),", Current slice number: ", string(i_slice));
            disp(mesg)

            %% retrieve the mean value of the current slice
            current_slice_octa_gt=squeeze(current_octa_gt(i_slice,:,:));
            mean_current_slice_octa_gt=mean(current_slice_octa_gt,"all");


            %% start manual denoise for this slice
            % if the current mean is not zero then start the denoise
            if mean_current_slice_octa_gt>0
    
                % setup mannual selection window for gap filling
                cropped_reg={}; % store cropped region
                xy_ind={}; % store the x and y indices of the cropped windows
    
    
                %% launch image cropping
                % selection window z
                img_title="OCTA groundtruth (Positive binarization)";
                img_note="Please press g to start, and q for exit";
                octa_fig=figure;  cla;
                imshow(current_slice_octa_gt);
                title(img_title);
                xlabel(img_note);
    
                % user interface for rectangles roi selection
                while true
                    % Get the current key
                    key=waitforbuttonpress;
                    currentKey = get(gcf, 'CurrentCharacter');
    
                    % if the 'g' key was pressed to crop
                    if currentKey == 'g'
                        % Allow user to draw a rectangle (on the first image)
                        hRect = drawrectangle;
                        position=hRect.Position;
    
                        % Wait for the rectangle to be finalized
                        wait(hRect);
                        rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 2);
    
                        % Optional: Highlight the cropped region on the image
                        % If the rectangle is empty, continue to next iteration
                        if isempty(hRect)
                            continue;
                        end
    
                        % Add the rectangle position to the list
                        cropped_reg{end+1} = position;
                        % Translate position to indices
                        x_start = round(position(1));
                        y_start = round(position(2));
                        width = round(position(3));
                        height = round(position(4));
                        x_end =x_start+width-1;
                        y_end =y_start+height-1;
                        x_crop_ind=[x_start;x_end];
                        y_crop_ind=[y_start;y_end];
                        xy_crop_ind=[x_crop_ind,y_crop_ind];
                        xy_ind{end+1}=xy_crop_ind;
    
                    % Check if the 'q' key was pressed to quit
                    elseif currentKey == 'q'
                        hold off
                        disp('Exiting cropping mode.');
                        % grab out the cropped missing region (launch this part only when
                        % there exists a cropped region)
                        num_elements=size(xy_ind,2);
    
                        % if the current slice contains regions for denoising,
                        % conduct denoise for the selected region
                        if num_elements~=0
    
                            % grab out each cropped region and denoise the selected position
                            for i_slice_crop=1:num_elements
                                current_crop_ind=xy_ind{i_slice_crop};
                                current_crop_ind_x=current_crop_ind(:,1);
                                current_crop_ind_y=current_crop_ind(:,2);
                                current_slice_octa_gt(current_crop_ind_y(1):current_crop_ind_y(2),current_crop_ind_x(1):current_crop_ind_x(2))=0;
                            end % ending for slice crop
                            
                            % store in current temporary variable
                            example_octa_gt_img(i_slice,:,:)=current_slice_octa_gt;
    
                            % close current figure window
                            close(octa_fig);
                            
                            % if this reaches the final slice of current
                            % sample, store it back
                            if i_slice==num_slice
                                den_octa_gt_storage{i_file}=example_octa_gt_img;
                                example_octa_gt_img=den_octa_gt_storage{i_file+1};
                                start_slice=1;
                            end
                            break;
                        
                        % if the current slice contains no region to denoise,
                        % pass this slice
                        else
                            close(octa_fig);
                            break;
                        end % ending for num_element~=0
    
                    % if the input is not g nor q terminate the entire function (temporarily exit mode)
                    else
                        % display confirmation message
                        mesg=strcat("Do you really wish to stop? Prese any key to resume Press q to stop");
                        disp(mesg);
                        
                        % wait for confirmation button
                        key=waitforbuttonpress;
                        currentKey = get(gcf, 'CurrentCharacter');

                        if currentKey=='q'
                            % check if the intermediate directory is created
                            if ~isfolder("Phase_1_Segmentation/data refiner function/temp")
                                mkdir("Phase_1_Segmentation/data refiner function/temp")
                            end
        
                            % save intermediate file
                            save("Phase_1_Segmentation/data refiner function/temp/den_octa_data_gt.mat","den_octa_gt_storage","-v7.3");
                            save("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_temp.mat","example_octa_gt_img");
                            save("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_slice.mat","i_slice");
                            save("Phase_1_Segmentation/data refiner function/temp/octa_data_gt_file.mat","i_file");
                            return;
                        end
    
                    end % ending for key 'g'
    
                end % ending for while true

            end % if for mean
    
        end % ending for i_slice_z 

    end % ending for i_file

end
