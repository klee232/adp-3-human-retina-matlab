function [filtered_octa_img_gt]=image_groundtruth_refiner_octa_negative(filtered_octa_img_gt,current_oct_img)
    % loop through every slice
    for i_slice=start_ind:end_ind
        disp(string(i_slice))
        current_series=squeeze(filtered_octa_img_gt(i_slice,:,:));
        current_series_oct=squeeze(current_oct_img(i_slice,:,:));
        % setup mannual selection window for gap filling
        cropped_reg={}; % store cropped region
        xy_ind={}; % store the x and y indices of the cropped windows
        % launch image cropping
        % selection window z
        img_title="OCTA groundtruth (Positive binarization)";
        img_note="Please press g to start, and q for exit";
        figure;  cla;
        octa_fig=subplot(121); cla;
        imshow(current_series);
        title(img_title);
        xlabel(img_note);
        img_title="OCT raw image";
        oct_fig=subplot(122); cla;
        imshow(current_series_oct);
        title(img_title);
        while true
            % Get the current key
            key=waitforbuttonpress;
            currentKey = get(gcf, 'CurrentCharacter');
            % Check if the 'g' key was pressed to crop
            if currentKey == 'g'
                % Allow user to draw a rectangle (on the first image)
                axes(octa_fig);
                hRect = drawrectangle(octa_fig);
                position=hRect.Position;
                % Wait for the rectangle to be finalized
                wait(hRect);
                axes(octa_fig);
                rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 2);
                % show the rectangle (on the second image)
                axes(oct_fig);
                hold on;
                rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 2);
                hold on;
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
                if num_elements~=0
                    % grab out each cropped region and fill the missing position
                    for i_slice_crop=1:num_elements
                        clickCoordinates_disp=[];
                        clickCoordinates=[];
                        position=cropped_reg{i_slice_crop};
                        current_crop_ind=xy_ind{i_slice_crop};
                        current_crop_ind_x=current_crop_ind(:,1);
                        current_crop_ind_y=current_crop_ind(:,2);
                        cropped_reg_img=imcrop(current_series,position);
                        cropped_reg_img_oct=imcrop(current_series_oct,position);
                        % (bulk clicking part) start the manually
                        % selection window for filling missing points
                        % (this is done by drawing rectangular windows)
                        figure; cla;
                        crop_octa=subplot(121); cla;
                        imshow(cropped_reg_img);
                        title("cropped octa image");
                        xlabel('Draw rectnagles to fill missing points. Press g to start. Press q if you want to exit');
                        crop_oct=subplot(122); cla;
                        imshow(cropped_reg_img_oct);
                        title("cropped oct image");
                        while true
                            key=waitforbuttonpress;
                            % Get the current key
                            currentKey = get(gcf, 'CurrentCharacter');
                            if currentKey == 'g'
                                % Allow user to draw a rectangle
                                hRect = drawrectangle(crop_octa);
                                position=hRect.Position;
                                % Wait for the rectangle to be finalized
                                wait(hRect);
                                % Optional: Highlight the cropped region on the image
                                rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 1);
                                axes(crop_oct);
                                hold on;
                                rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 1);
                                hold off
                                % If the rectangle is empty, continue to next iteration
                                if isempty(hRect)
                                    continue;
                                end
                                % Add the rectangle position to the list
                                % Translate position to indices
                                x_start_bulk = round(position(1));
                                y_start_bulk = round(position(2));
                                width_bulk = round(position(3));
                                height_bulk = round(position(4));
                                x_end_bulk =x_start_bulk+width_bulk-1;
                                y_end_bulk =y_start_bulk+height_bulk-1;
                                x_crop_ind_bulk=[x_start_bulk;x_end_bulk];
                                y_crop_ind_bulk=[y_start_bulk;y_end_bulk];
                                % change the region to all 1s
                                cropped_reg_img(y_crop_ind_bulk(1):y_crop_ind_bulk(2),x_crop_ind_bulk(1):x_crop_ind_bulk(2))=1;
                                % if q is entered exit current mode
                            elseif currentKey == 'q'
                                figure; cla;
                                subplot(121);
                                imshow(cropped_reg_img);
                                title('Current outcome. Press g to keep refining. Press q to enter next stage');
                                subplot(122);
                                imshow(cropped_reg_img_oct);
                                title("corresponding oct crop");
                                key=waitforbuttonpress;
                                % Get the current key
                                currentKey = get(gcf, 'CurrentCharacter');
                                if currentKey=='q'
                                    break;
                                end
                            end
                        end % ending for while true
        
                        % (detail clicking part) start the manually selection window for filling missing
                        % points (this is done by clicking the pixel position by
                        % mouse click)
                        figure;
                        subplot(121);
                        imshow(cropped_reg_img);
                        title('Click missing points. Press q if you want to exit');
                        subplot(122);
                        imshow(cropped_reg_img_oct);
                        title('corresponding oct raw image');
                        while true
                            % Wait for a mouse click or key press
                            [x, y, button] = ginput(1);
                            if x<=0 || y<=0
                                [row_ind,col_ind]=find(squeeze(cropped_reg_img));
                                disp_x=round(row_ind(1));
                                disp_y=round(col_ind(1));
                            else
                                disp_x=round(x);
                                disp_y=round(y);
                            end
                            real_x=round(x+current_crop_ind_x(1));
                            real_y=round(y+current_crop_ind_y(1));
                            % Store the click coordinates if mouse click
                            if button == 1 % Left mouse button click
                                clickCoordinates_disp=[clickCoordinates_disp;disp_x, disp_y];
                                clickCoordinates=[clickCoordinates; real_x, real_y];
                                % Plot the clicked point
                                hold on;
                                plot(disp_x, disp_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
                            end
                            % Check if 'q' key is pressed (ASCII code for 'q' is 113)
                            if button == 113  % 'q' key
                                hold off
                                % check the click number
                                num_clickCoords=size(clickCoordinates,1);
                                if isempty(clickCoordinates)
                                    disp('Exiting selection mode.');
                                    break;
                                else
                                    for i_clickCoords=1:num_clickCoords
                                        current_coord=clickCoordinates(i_clickCoords,:);
                                        current_coord_disp=clickCoordinates_disp(i_clickCoords,:);
                                        current_series(current_coord(2),current_coord(1))=1;
                                        cropped_reg_img(current_coord_disp(2),current_coord_disp(1))=1;
                                    end
                                    % store current series back to outcome
                                    filtered_octa_img_gt(i_slice,:,:)=current_series;
                                    % show up the manual selection outcome
                                    figure;
                                    subplot(121);
                                    imshow(cropped_reg_img);
                                    title("Improvement Outcome. Press g to refine again. Press q to quit.")
                                    subplot(122);
                                    imshow(cropped_reg_img_oct);
                                    title("corresponding oct raw image");
                                    key=waitforbuttonpress;
                                    % Get the current key
                                    currentKey = get(gcf, 'CurrentCharacter');
                                    % if q is pressed again, exit this loop
                                    if currentKey == 'q'
                                        disp('Exiting selection mode.');
                                        break;
                                    end
                                end % ending for if isempty
                            end % ending for button==113
                        end % ending for while true
        
                        % display final result
                        if i_slice_crop==num_elements
                            figure;
                            imshow(current_series);
                            title("Current Slice Outcome");
                            % save intermediate file
                            save("temp_data/octa_data_gt_series.mat","filtered_octa_img_gt");
                            save("temp_data/octa_data_gt_ind.mat","i_slice")
                        end
                    end % ending for slice crop
                    break;
                else
                    break;
                end % ending for num_element~=0
            else % if the input is not g nor q terminate the entire function (temporarily exit mode)
                % save intermediate file
                save("temp_data/octa_data_gt_series.mat","filtered_octa_img_gt");
                save("temp_data/octa_data_gt_ind.mat","i_slice")
                return;
            end % ending for key 'g'
        end % ending for while true
    end% ending for i_slice_z  
end