function [filtered_octa_img_gt]=image_groundtruth_refiner_octa_positive(filtered_octa_img_gt,current_oct_img)
    % loop through every slice
    disp(string(i_slice))
    current_series=squeeze(filtered_octa_img_gt(i_slice,:,:));
    current_series_oct=squeeze(current_oct_img(i_slice,:,:));
    % launch positive clicking window
    [current_series]=image_groundtruth_refiner_octa_pos(current_series,current_series_oct);

        
end