% Created by Kuan-Min Lee
% Createed date: Jan. 28th, 2025

% Brief User Introduction:
% This function is built to visualize the vessel length in 3d image

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]
% 4D image array format: [num_file, num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% data_image: vessel length variable created in vessel length calculator
% function


function analyzer_color_visualizer(data_image)

    %% grab out the maximum and minimum value to create color bar
    max_intensity=max(data_image,[],'all');
    min_intensity=min(data_image,[],"all");


    %% create color figure windwo for 3d image
    figure;
    h=volshow(data_image,'Colormap', jet);
    h.Parent.CLim = [min_intensity max_intensity];

    % Add a colorbar to the figure
    colorbar;

    % Set label for the colorbar
    c = colorbar;
    c.Label.String = 'Intensity';
    

end