

function [close_data_storage]=image_groundtruth_generator_en_face_close(data_storage)

    %% conduct image closing on each image
    num_file=size(data_storage,1);
    se=strel('rectangle',[5 5]);
    se_connect=fspecial('disk',5);

    close_data_storage=zeros(size(data_storage));
    for i_file=1:num_file
        current_data_img=squeeze(data_storage(i_file,:,:));
        close_current_data_img=imclose(current_data_img,se);
        close_current_data_img=conv2(close_current_data_img,se_connect,'same');
        close_data_storage(i_file,:,:)=close_current_data_img;
    end
    
    
    %% conduct refining binarization on each image
    for i_file=1:num_file
        close_current_data_img=squeeze(close_data_storage(i_file,:,:));
        close_current_data_img_mean=mean2(close_current_data_img);
        close_current_data_img_std=std2(close_current_data_img);

        % apply image contrast with higher threshold of mean+2*std and
        % lower threshold of mean-2*std
        lower_threshold=max([(close_current_data_img_mean-2*close_current_data_img_std) 0]);
        higher_threshold=min([(close_current_data_img_mean+2*close_current_data_img_std) 1]);
        close_current_data_img_contrast=imadjust(close_current_data_img, [lower_threshold higher_threshold],[]);

        % enhance the image contrast again using contrast limited adaptive
        % histogram equalization
        close_current_data_img_clahe=adapthisteq(close_current_data_img_contrast, 'ClipLimit', 0.05, 'NumTiles', [8 8]);

        % binarized the image
        refine_data_img=imbinarize(close_current_data_img_clahe,0.7);
        close_data_storage(i_file,:,:)=refine_data_img;
    end

end