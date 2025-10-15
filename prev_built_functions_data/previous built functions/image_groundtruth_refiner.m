

function [filtered_octa_img_gt, filtered_oct_img_gt, picture_obj]=image_groundtruth_refiner(picture_obj,current_octa_img_gt,current_oct_img_gt)
    % create outcome storage variable 
    filtered_octa_img_gt=current_octa_img_gt;
    filtered_oct_img_gt=current_oct_img_gt;

    % grab out only the region that contains content based on the previous
    % manually selected region
    z_ind=picture_obj.z_ind;

    % loop through every slice
    for i_slice=z_ind(1):z_ind(2)
        current_series=squeeze(current_octa_img_gt(i_slice,:,:));
        % setup mannual selection window for gap filling
        cropped_reg={}; % store cropped region
        xy_ind={}; % store the x and y indices of the cropped windows          
        % launch image cropping
        % selection window z
        img_title=strcat("Please press g to start, and q for exit");
        figure(1);  cla;
        imshow(current_series);
        title(img_title);
        while true
            key=waitforbuttonpress;
            % Get the current key
            currentKey = get(gcf, 'CurrentCharacter');
            % Check if the 'q' key was pressed to quit
            if currentKey == 'q'
                disp('Exiting cropping mode.');
                break;
            end
            % Check if the 'g' key was pressed to crop
            if currentKey == 'g'
                % Allow user to draw a rectangle
                hRect = drawrectangle;
                position=hRect.Position;
                % Wait for the rectangle to be finalized
                wait(hRect);
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
                % Optional: Highlight the cropped region on the image
                rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 2);
            end
        end
        % grab out the cropped missing region (launch this part only when
        % there exists a cropped region)
        num_elements=size(xy_ind,1);
        if num_elements~=0
            % grab out each cropped region and fill the missing position
            for i_slice_crop=1:num_elements
                current_crop_ind=xy_ind{i_slice_crop};
                current_crop_ind_x=current_crop_ind(:,1);
                current_crop_ind_y=current_crop_ind(:,2);
                cropped_reg_img=imcrop(current_series,position);
                % start the manually selection window for filling missing
                % points (this is done by clicking the pixel position by
                % mouse click)
                clickCoordinates=[];
                figure;
                imshow(cropped_reg_img);
                title('Click missing points. Press q if you want to exit');
                while true
                    % Wait for a mouse click or key press
                    [x, y, button] = ginput(1);
                    disp_x=x;
                    disp_y=y;
                    x=round(x+current_crop_ind_x(1));
                    y=round(y+current_crop_ind_y(1));
                    % Check if 'q' key is pressed (ASCII code for 'q' is 113)
                    if button == 113  % 'q' key
                        disp('Exiting selection mode.');
                        break;
                    end
                    % Store the click coordinates if mouse click
                    if button == 1 % Left mouse button click
                        clickCoordinates = [clickCoordinates; x, y];             
                        % Plot the clicked point
                        hold on;
                        plot(disp_x, disp_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
                    end
                end
            end
        end
        % assign 1 to the stored click coordinates
        num_clickCoords=size(clickCoordinates,1);
        for i_clickCoords=1:num_clickCoords
            current_coord=clickCoordinates(i_clickCoords,:);
            current_series(current_coord(1),current_coord(2))=1;
        end
        % store current series back to outcome
        filtered_octa_img_gt(i_slice,:,:)=current_series;
     end
 end

