function data_distributor()
    % load data 
    octa_gt_temp=load("temp_data/octa_data_gt_series.mat");
    oct_gt_temp=load("temp_data/oct_data_gt_series.mat");
    picture_obj_octa=load("processed_data/picture_obj_octa.mat");
    picture_obj_oct=load("processed_data/picture_obj_oct.mat");

    % make new directory for storage
    mkdir("data_for_distribution")

    octa_gt_temp=octa_gt_temp.filtered_octa_img_gt;
    oct_gt_temp=oct_gt_temp.filtered_oct_img_gt;
    picture_obj_octa=picture_obj_octa.picture_obj_octa;
    picture_obj_oct=picture_obj_oct.picture_obj_oct;

    % conduct image partition
    picture_obj_octa.z_ind(2)=(picture_obj_octa.z_ind(2)+picture_obj_octa.z_ind(1))/2;
    picture_obj_oct.z_ind(2)=(picture_obj_oct.z_ind(2)+picture_obj_oct.z_ind(1))/2;
    octa_gt_part=octa_gt_temp(1:picture_obj_octa.z_ind(2),:,:);
    oct_gt_part=oct_gt_temp(1:picture_obj_oct.z_ind(2),:,:);

    % save partitioned data
    save('data_for_distribution/octa_gt.mat',"octa_gt_part")
    save('data_for_distribution/oct_gt.mat',"oct_gt_part")
    save('data_for_distribution/picture_obj_octa.mat',"picture_obj_octa");
    save('data_for_distribution/picture_obj_oct.mat',"picture_obj_oct");
end