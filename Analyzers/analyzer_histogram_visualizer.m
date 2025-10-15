% Created by Kuan-Min Lee
% Createed date: Jan. 28th, 2025

% Brief User Introduction:
% This function is built to visualize the vessel length distribution in 3d image

% 3D image array format: [num_z_slice, num_x_coord, num_y_coord]
% 4D image array format: [num_file, num_z_slice, num_x_coord, num_y_coord]

% Input parameter:
% data_image: vessel length variable created in vessel length calculator
% function


function analyzer_histogram_visualizer(data_image)

    %% compute histogram of current image
    figure;
    histogram(data_image(find(data_image)));
    grid on


end