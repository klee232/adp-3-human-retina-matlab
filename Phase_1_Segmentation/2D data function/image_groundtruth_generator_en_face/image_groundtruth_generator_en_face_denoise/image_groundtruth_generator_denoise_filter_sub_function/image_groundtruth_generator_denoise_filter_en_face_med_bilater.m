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


function denoised_data_storage=image_groundtruth_generator_denoise_filter_en_face_med_bilater(data_stroage, region_size, patch_var, spatial_sigma)
    
    %% conduct filtering for each slice
    denoised_data_storage=zeros(size(data_stroage));
    num_files=size(data_stroage,1);
    for i_file=1:num_files
        current_data_img=data_stroage(i_file,:,:);
        current_data_img=squeeze(current_data_img);


        %% conduct median filtering with given region size
        current_data_img=medfilt2(current_data_img,region_size);


        %% conduct bilateral filtering
        current_data_img=imbilatfilt(current_data_img,patch_var,spatial_sigma);


        %% store the final result
        denoised_data_storage(i_file,:,:)= current_data_img;

    end
end
