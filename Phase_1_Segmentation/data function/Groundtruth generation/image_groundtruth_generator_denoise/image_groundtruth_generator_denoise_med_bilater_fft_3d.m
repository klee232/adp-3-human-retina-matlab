% Created by Kuan-Min Lee
% Createed date: Jan. 9th, 2024

% Brief User Introduction:
% The following function is utilized to perform image denoising for choroid
% layer specifically. The process contains as follow: 1. median filtering
% 2. bilateral filtering, and 3. fast fourier transform

% input argument:
% input_volume: 3D array of input volume (in our case, 3D image for choroid
% layer)
% region_size: size of the sub-volumes for regional processing, [x, y, z]

% output:
% enhanced_volume: output 3D array with enhanced contrast


function enhanced_volume=image_groundtruth_generator_denoise_med_bilater_fft_3d(input_volume, region_size)
    
    %% conduct filtering for each slice
    enhanced_volume=zeros(size(input_volume));
    num_slice=size(input_volume,1);
    for i_slice=1:num_slice
        current_slice=input_volume(i_slice,:,:);
        current_slice=squeeze(current_slice);


        %% conduct median filtering with given region size
        current_slice=medfilt2(current_slice,region_size);


        %% conduct bilateral filtering
        current_slice=imbilatfilt(current_slice,10,1);


        %% conduct fast fourier transformation filtering
        % conduct fast fourier transform
        f_current_slice=fft2(current_slice);
        f_current_slice=fftshift(f_current_slice);
        [rows, cols] = size(f_current_slice);
        crow = round(rows/2);
        ccol = round(cols/2);

        % conduct low-pass filtering. since the noise is typically in
        % higher frequency, we keep only the low-frequency feature
        radius = 30; % Low-pass filter radius
        [x, y] = meshgrid(1:cols, 1:rows);
        low_pass_filter = sqrt((x-ccol).^2 + (y-crow).^2) <= radius;
        F_filtered = f_current_slice .* low_pass_filter;


        %% store the final result
        enhanced_volume(i_slice,:,:)= abs(ifft2(ifftshift(F_filtered)));
    end
end
