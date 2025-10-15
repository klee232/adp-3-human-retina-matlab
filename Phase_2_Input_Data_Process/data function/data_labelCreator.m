% Created by Kuan-Min Lee
% Created date: Oct. 30th, 2024
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to data partitioning for the training of our OCTA network for Alzheimer's Disease 

% Input Parameter
% none

% Output Parameter
% label_gt_storage: label groundtruth storage variable


function [label_gt_storage]=data_labelCreator()
    
    %% read the data table
    data_tbl=readtable("~/data/klee232/processed_data/ADP-3 data.xlsx");

    %% create label for data based on data subject id
    % create label storage variable
    label_gt_storage=categorical(strings(68,1));
    % read picture object folder to retrieve data name
    all_picture_obj=dir('~/data/klee232/processed_data/picture_obj/*-OCTA_obj.mat');
    num_picture_obj=length(all_picture_obj);
    for i_picture_obj=1:num_picture_obj
        current_picture_obj=all_picture_obj(i_picture_obj).name;
        % grab out sample id
        current_sample_id=current_picture_obj(10:17);
        % find out the corresponding group
        current_sample_row=data_tbl(ismember(data_tbl.SubjectID,current_sample_id), :);
        current_sample_label=current_sample_row.Group;
        % store the corresponding label
        label_gt_storage(i_picture_obj,1)=current_sample_label;
    end

end