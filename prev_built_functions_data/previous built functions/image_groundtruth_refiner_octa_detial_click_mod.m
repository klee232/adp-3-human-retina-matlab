% (detail clicking part)start the manually selection window for filling missing
% points (this is done by clicking the pixel position by
% mouse click)
figure(3);
imshow(cropped_reg_img);
title('Click missing points. Press q if you want to exit');
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
            figure(3);
            imshow(cropped_reg_img);
            title("Improvement Outcome. Press g to refine again. Press q to quit.")
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