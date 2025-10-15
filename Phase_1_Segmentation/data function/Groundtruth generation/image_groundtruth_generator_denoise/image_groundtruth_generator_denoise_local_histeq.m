function enhanced_image = image_groundtruth_generator_denoise_local_histeq(image, window_size)
    % Local Histogram Equalization
    % image: input 2D grayscale image
    % window_size: size of the local region (e.g., [8, 8])
    % enhanced_image: output image after local histogram equalization

    % Get the image size
    [sx, sy, sz] = size(image);
    
    % Pad the image to handle borders
    pad_size = floor(window_size / 2);
    padded_image=zeros(sx,(sy+2*pad_size(1)),(sz+2*pad_size(2)));
    for x=1:sx
        current_image=image(x,:,:);
        current_image=squeeze(current_image);
        padded_current_image=padarray(current_image, pad_size, 'symmetric');
        padded_image(x,:,:)=padded_current_image;
    end

    % Initialize the enhanced image
    enhanced_image = zeros(size(image));
    
    % Process each pixel with local histogram equalization
    mean_store=mean(image,[2 3]);

    for x = 1:sx
        current_mean_pixel=mean_store(x,1);
        if current_mean_pixel>0.06
            for y = 1:sy
                for z=1:sz
                    % Define the local region (window)
                    y_start = y;
                    y_end = y + window_size(1) - 1;
                    z_start = z;
                    z_end = z + window_size(2) - 1;
                    
                    % Extract the local region
                    local_region = padded_image(x,y_start:y_end, z_start:z_end);
                    local_region = squeeze(local_region);
                    
                    % Apply histogram equalization to the local region
                    equalized_region = histeq(local_region);
                    
                    % Set the center pixel of the local region to the equalized value
                    enhanced_image(x, y, z) = equalized_region(pad_size(1) + 1, pad_size(2) + 1);
                end
            end
        end
    end
end