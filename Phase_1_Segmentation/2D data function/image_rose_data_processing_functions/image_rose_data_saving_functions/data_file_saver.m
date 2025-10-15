% Created by Kuan-Min Lee
% Created date: Dec. 13th, 2023 (last updated: Jan. 10th, 2024)
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is created to save the image arrays as .mat files (for convenience of later usage) 

% Input Parameter: 
% all dataset that have been used (numerical arrays)
% Output Parameter:
% None

function data_file_saver(train_ROSE_SVC_org, valid_ROSE_SVC_org, ...
                        train_ROSE_SVC_orgGt, valid_ROSE_SVC_orgGt, ...
                        train_ROSE_SVC_thinGt, valid_ROSE_SVC_thinGt, ...
                        train_ROSE_SVC_thickGt, valid_ROSE_SVC_thickGt, ...
                        test_ROSE_SVC_org, test_ROSE_SVC_orgGt, test_ROSE_SVC_thinGt, test_ROSE_SVC_thickGt,...
                        train_ROSE_DVC_org, valid_ROSE_DVC_org, ...
                        train_ROSE_DVC_orgGt, valid_ROSE_DVC_orgGt,...
                        test_ROSE_DVC_org, test_ROSE_DVC_orgGt, ...
                        train_ROSE_SDVC_org, valid_ROSE_SDVC_org, ...
                        train_ROSE_SDVC_orgGt, valid_ROSE_SDVC_orgGt, ...
                        test_ROSE_SDVC_org, test_ROSE_SDVC_orgGt, ...
                        train_ROSE2_org, valid_ROSE2_org, ...
                        train_ROSE2_orgGt, valid_ROSE2_orgGt,...
                        test_ROSE2_org, test_ROSE2_orgGt)
    
    % Save File Section (for later usage)
    % Check if the files are already save
    indicator=isfolder("train");
    % if so, jump out the program
    if indicator
        disp("All file have been saved before.")
        return

    % if not save all the files
    else
        % Create Directory for Data storage
        main_data_save_path="C:\Users\klee232\Desktop\Thesis\Codes\My works\";
        % training dataset part
        mkdir train
        train_data_save_path=strcat(main_data_save_path,"train\");
        train_ROSE_SVC_org_filename=strcat(train_data_save_path,"train_ROSE_SVC_org.mat");
        valid_ROSE_SVC_org_filename=strcat(train_data_save_path,"valid_ROSE_SVC_org.mat");
        train_ROSE_SVC_orgGt_filename=strcat(train_data_save_path,"train_ROSE_SVC_orgGt.mat");
        valid_ROSE_SVC_orgGt_filename=strcat(train_data_save_path,"valid_ROSE_SVC_orgGt.mat");
        train_ROSE_SVC_thinGt_filename=strcat(train_data_save_path,"train_ROSE_SVC_thinGt.mat");
        valid_ROSE_SVC_thinGt_filename=strcat(train_data_save_path,"valid_ROSE_SVC_thinGt.mat");
        train_ROSE_SVC_thickGt_filename=strcat(train_data_save_path,"train_ROSE_SVC_thickGt.mat");
        valid_ROSE_SVC_thickGt_filename=strcat(train_data_save_path,"valid_ROSE_SVC_thickGt.mat");
        train_ROSE_DVC_org_filename=strcat(train_data_save_path,"train_ROSE_DVC_org.mat");
        valid_ROSE_DVC_org_filename=strcat(train_data_save_path,"valid_ROSE_DVC_org.mat");
        train_ROSE_DVC_orgGt_filename=strcat(train_data_save_path,"train_ROSE_DVC_orgGt.mat"); 
        valid_ROSE_DVC_orgGt_filename=strcat(train_data_save_path,"valid_ROSE_DVC_orgGt.mat"); 
        train_ROSE_SDVC_org_filename=strcat(train_data_save_path,"train_ROSE_SDVC_org.mat");
        valid_ROSE_SDVC_org_filename=strcat(train_data_save_path,"valid_ROSE_SDVC_org.mat");
        train_ROSE_SDVC_orgGt_filename=strcat(train_data_save_path,"train_ROSE_SDVC_orgGt.mat");
        valid_ROSE_SDVC_orgGt_filename=strcat(train_data_save_path,"valid_ROSE_SDVC_orgGt.mat");
        train_ROSE2_org_filename=strcat(train_data_save_path,"train_ROSE2_org.mat");
        valid_ROSE2_org_filename=strcat(train_data_save_path,"valid_ROSE2_org.mat");
        train_ROSE2_orgGt_filename=strcat(train_data_save_path,"train_ROSE2_orgGt.mat");
        valid_ROSE2_orgGt_filename=strcat(train_data_save_path,"valid_ROSE2_orgGt.mat");
%         train_ROSEO_org_filename=strcat(train_data_save_path,"train_ROSEO_org.mat");
%         valid_ROSEO_org_filename=strcat(train_data_save_path,"valid_ROSEO_org.mat");
%         train_ROSEO_orgGt_filename=strcat(train_data_save_path,"train_ROSEO_orgGt.mat");
%         valid_ROSEO_orgGt_filename=strcat(train_data_save_path,"valid_ROSEO_orgGt.mat");
        save(train_ROSE_SVC_org_filename,"train_ROSE_SVC_org");
        save(valid_ROSE_SVC_org_filename,"valid_ROSE_SVC_org");
        save(train_ROSE_SVC_orgGt_filename,"train_ROSE_SVC_orgGt");
        save(valid_ROSE_SVC_orgGt_filename,"valid_ROSE_SVC_orgGt");
        save(train_ROSE_SVC_thinGt_filename,"train_ROSE_SVC_thinGt");
        save(valid_ROSE_SVC_thinGt_filename,"valid_ROSE_SVC_thinGt");
        save(train_ROSE_SVC_thickGt_filename,"train_ROSE_SVC_thickGt");
        save(valid_ROSE_SVC_thickGt_filename,"valid_ROSE_SVC_thickGt");
        save(train_ROSE_DVC_org_filename,"train_ROSE_DVC_org");
        save(valid_ROSE_DVC_org_filename,"valid_ROSE_DVC_org");
        save(train_ROSE_DVC_orgGt_filename,"train_ROSE_DVC_orgGt");
        save(valid_ROSE_DVC_orgGt_filename,"valid_ROSE_DVC_orgGt");
        save(train_ROSE_SDVC_org_filename,"train_ROSE_SDVC_org");
        save(valid_ROSE_SDVC_org_filename,"valid_ROSE_SDVC_org");
        save(train_ROSE_SDVC_orgGt_filename,"train_ROSE_SDVC_orgGt");
        save(valid_ROSE_SDVC_orgGt_filename,"valid_ROSE_SDVC_orgGt");
        save(train_ROSE2_org_filename,"train_ROSE2_org");
        save(valid_ROSE2_org_filename,"valid_ROSE2_org");
        save(train_ROSE2_orgGt_filename,"train_ROSE2_orgGt");
        save(valid_ROSE2_orgGt_filename,"valid_ROSE2_orgGt");
%         save(train_ROSEO_org_filename,"train_ROSEO_org");
%         save(valid_ROSEO_org_filename,"valid_ROSEO_org");
%         save(train_ROSEO_orgGt_filename,"train_ROSEO_orgGt");
%         save(valid_ROSEO_orgGt_filename,"valid_ROSEO_orgGt");

    
        % testing dataset part
        mkdir test
        test_data_save_path=strcat(main_data_save_path,"test\");
        test_ROSE_SVC_org_filename=strcat(test_data_save_path,"test_ROSE_SVC_org.mat");
        test_ROSE_SVC_orgGt_filename=strcat(test_data_save_path,"test_ROSE_SVC_orgGt.mat");
        test_ROSE_SVC_thinGt_filename=strcat(test_data_save_path,"test_ROSE_SVC_thinGt.mat");
        test_ROSE_SVC_thickGt_filename=strcat(test_data_save_path,"test_ROSE_SVC_thickGt.mat");
        test_ROSE_DVC_org_filename=strcat(test_data_save_path,"test_ROSE_DVC_org.mat");
        test_ROSE_DVC_orgGt_filename=strcat(test_data_save_path,"test_ROSE_DVC_orgGt.mat"); 
        test_ROSE_SDVC_org_filename=strcat(test_data_save_path,"test_ROSE_SDVC_org.mat");
        test_ROSE_SDVC_orgGt_filename=strcat(test_data_save_path,"test_ROSE_SDVC_orgGt.mat");
        test_ROSE2_org_filename=strcat(test_data_save_path,"test_ROSE2_org.mat");
        test_ROSE2_orgGt_filename=strcat(test_data_save_path,"test_ROSE2_orgGt.mat");
%         test_ROSEO_org_filename=strcat(test_data_save_path,"test_ROSEO_org.mat");
%         test_ROSEO_orgGt_filename=strcat(test_data_save_path,"test_ROSEO_orgGt.mat");
        save(test_ROSE_SVC_org_filename,"test_ROSE_SVC_org");
        save(test_ROSE_SVC_orgGt_filename,"test_ROSE_SVC_orgGt");
        save(test_ROSE_SVC_thinGt_filename,"test_ROSE_SVC_thinGt");
        save(test_ROSE_SVC_thickGt_filename,"test_ROSE_SVC_thickGt");
        save(test_ROSE_DVC_org_filename,"test_ROSE_DVC_org");
        save(test_ROSE_DVC_orgGt_filename,"test_ROSE_DVC_orgGt");
        save(test_ROSE_SDVC_org_filename,"test_ROSE_SDVC_org");
        save(test_ROSE_SDVC_orgGt_filename,"test_ROSE_SDVC_orgGt");
        save(test_ROSE2_org_filename,"test_ROSE2_org");
        save(test_ROSE2_orgGt_filename,"test_ROSE2_orgGt");
%         save(test_ROSEO_org_filename,"test_ROSEO_org");
%         save(test_ROSEO_orgGt_filename,"test_ROSEO_orgGt");

    end

end