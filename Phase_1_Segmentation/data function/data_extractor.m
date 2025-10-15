% Created by Kuan-Min Lee
% Created date: Jan. 6th, 2025
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is created to extract and save all the files originally
% stored in the octa_gt_storage file


%% load all stored files
load('~/data/klee232/processed_data/pad_octa_gt_data.mat');
load('~/data/klee232/processed_data/pad_octa_data.mat');

%% load and save files
file_dir='~/data/klee232/processed_data/data_for_refining/';
if ~isfolder(file_dir)
    mkdir ('~/data/klee232/processed_data/data_for_refining/')
end

num_files=size(octa_gt_storage,1);

for i_file=1:num_files
    % extract all stored image files 
    current_octa_storage=octa_storage{i_file};
    current_octa_gt_storage=octa_gt_storage{i_file};
    
    % save current image
    file_name=strcat('file_',string(i_file));
    org_file_name=strcat('org_file_',string(i_file));
    complete_file_name=strcat(file_dir,file_name,'.mat');
    complete_file_name_org=strcat(file_dir,org_file_name,'.mat');
    save(complete_file_name,"current_octa_gt_storage");
    save(complete_file_name_org,"current_octa_storage");
end


