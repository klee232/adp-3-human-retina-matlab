% Created by Kuan-Min Lee
% Createed date: Oct. 6th, 2024

% Brief User Introduction:
% The following function is utilized to perform regional dynamic histogram
% equalization for image contrast enhancement and denoise

% input argument:
% input_volume: 3D array of input volume (in our case, 3D image for choroid
% layer)
% region_size: size of the sub-volumes for regional processing, [x, y, z]

% output:
% enhanced_volume: output 3D array with enhanced contrast


function enhanced_volume = image_groundtruth_generator_denoise_rdhe_3d(input_volume, region_size)
    %% Get the size of the input volume
    [sx, sy, sz] = size(input_volume);

    % Initialize the output volume
    enhanced_volume = zeros(sx, sy, sz);

    
    %% create storage variable for mean subregion pixel value
    mean_sub_region_pixel=zeros(length(1:region_size(1):sx),...
                                length(1:region_size(2):sy),...
                                length(1:region_size(3):sz));


    %% Iterate over the volume in blocks of region_size
    mean_store=mean(input_volume,[2 3]);
    counter_x=1;
    counter_y=1;
    counter_z=1;
    for x = 1:region_size(1):sx
        if x==1
            counter_x=1;
        end
        for y = 1:region_size(2):sy
            if y==1
                counter_y=1;
            end
            for z = 1:region_size(3):sz
                if z==1
                    counter_z=1;
                end

                % Define the boundaries of the current region
                x_end = min(x + region_size(1) - 1, sx);
                y_end = min(y + region_size(2) - 1, sy);
                z_end = min(z + region_size(3) - 1, sz);

                % Extract the sub-volume (region)
                sub_volume = input_volume(x:x_end, y:y_end, z:z_end);

                % grab out current layer mean pixel value
                current_layer_mean_pixel=mean_store(x,1);

                % Apply histogram equalization to the sub-volume only if
                % the current pixel value is not zero
                if current_layer_mean_pixel>0
                    % calculate the mean pixel value for current subregion
                    current_subregion_mean_pixel=mean(sub_volume,'all');
                    mean_sub_region_pixel(counter_x,counter_y,counter_z)=current_subregion_mean_pixel;

                    sub_region_pixel_threshold=0.2;

                    if current_subregion_mean_pixel>sub_region_pixel_threshold
                        equalized_sub_volume = histeq3d(sub_volume);
                        % Place the enhanced sub-volume back into the output volume
                        enhanced_volume(x:x_end, y:y_end, z:z_end) = equalized_sub_volume;
                    end
                end
                counter_z=counter_z+1;
            end
            counter_y=counter_y+1;
        end
        counter_x=counter_x+1;
    end
end

function equalized_sub_volume = histeq3d(sub_volume)
    % Perform histogram equalization on a 3D sub-volume
    % sub_volume: 3D array representing a sub-volume
    % equalized_sub_volume: 3D array with enhanced contrast

    %% Flatten the 3D sub-volume into a 1D array
    flattened = sub_volume(:);


    %% Grab out the maximum and minimum of the sub region
    max_sub_volume=max(flattened);
    min_sub_volume=min(flattened);


    %% Perform histogram equalization on the flattened data
    equalized_flattened = histeq(flattened);


    %% Reshape the equalized data back into the original 3D shape
    equalized_sub_volume = reshape(equalized_flattened, size(sub_volume));
end
