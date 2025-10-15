% =======the below are the refiner part (not finished)============
        % load the indicator (this file contains indicator. It has only two value:
        % OCTA: indicates that the previous stopped file is OCTA file. 
        % OCT:indicates that the previous stopped file is OCT file.)
        % load("temp_data/indicator.mat");

        % if the last step stopped at octa, retrieve only octa groundtruth
        % file
        % if indicator=="octa"
        %     load("temp_data/octa_data_gt_series.mat"); % the variable inside is the filtered_octa_img_gt
        %     octa_gt_storage=zeros([num_octa_files size(filtered_octa_img_gt)]);
        %     octa_gt_storage(start_ind,:,:,:)=filtered_octa_img_gt;
        %     % oct_gt_storage=zeros([num_octa_files size(filtered_octa_img_gt)]);
        % elseif indicator=="oct"
        %     load("temp_data/octa_data_gt_series.mat");
        %     octa_gt_storage=zeros([num_octa_files size(filtered_octa_img_gt)]);
        %     octa_gt_storage(start_ind,:,:,:)=filtered_octa_img_gt; 
        %     % load("temp_data/oct_data_gt_series.mat");
        %     % oct_gt_storage=zeros([num_oct_files size(filtered_oct_img_gt)]);
        %     % oct_gt_storage(start_ind,:,:,:)=filtered_oct_img_gt;
        % end
        % ================================================================


        % conduct segmentation on refiner octa and oct
        % [current_octa_img_gt,current_oct_img_gt, picture_obj_octa, picture_obj_oct]=image_groundtruth_refiner_rev(picture_obj_octa,picture_obj_oct);
        % store back the refined groundtruth
        % octa_gt_storage(i_file,:,:,:)=current_octa_img_gt;
        % oct_gt_storage(i_file,:,:,:)=current_oct_img_gt;