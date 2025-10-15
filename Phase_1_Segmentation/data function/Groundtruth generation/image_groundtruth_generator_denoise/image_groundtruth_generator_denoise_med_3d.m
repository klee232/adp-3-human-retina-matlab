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


function enhanced_volume=image_groundtruth_generator_denoise_med_3d(input_volume, region_size)
    %% conduct median filtering for each slice
    enhanced_volume=zeros(size(input_volume));
    num_slice=size(input_volume,1);
    for i_slice=1:num_slice
        current_slice=input_volume(i_slice,:,:);
        current_slice=squeeze(current_slice);
        current_slice=imbilatfilt(current_slice,10,1);
        f_current_slice=fft2(current_slice);
        f_current_slice=fftshift(f_current_slice);
        [rows, cols] = size(f_current_slice);
        crow = round(rows/2);
        ccol = round(cols/2);
        radius = 30; % Low-pass filter radius
        [x, y] = meshgrid(1:cols, 1:rows);
        low_pass_filter = sqrt((x-ccol).^2 + (y-crow).^2) <= radius;
        F_filtered = f_current_slice .* low_pass_filter;
        enhanced_volume(i_slice,:,:)= abs(ifft2(ifftshift(F_filtered)));
    end


end
