% Created by Kuan-Min Lee
% Created date: May 21st, 2025
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is built to compute the testing criteria: pixel prediction
% accuracy (how many pixel of prediction is with the right classification)
% for the trained model

% Input Parameter:
% trained_coarse_network: the best trained coarse stage network from
% previous phase
% trained_fine_network: the best trained fine stage network from previous
% phase
% test_data: testing octa image
% test_gt_data: testing groundtruth image
% lyr: layer tested


function testing_function_pixel_accuracy(trained_coarse_network, trained_fine_network, test_data, test_gt_data, lyr)

    %% retrieve the test data dimensional information
    row_size=size(test_data,1);
    col_size=size(test_data,2);
    num_test_data=size(test_data,3);


    %% compute prediction and grab out groundtruth
    total_error=0;

    for i_test_data=1:num_test_data
        test_img=test_data(:,:,i_test_data);
        test_gt=test_gt_data(:,:,i_test_data);
        test_img=dlarray(test_img,"SSCB");

        % compute the outcome
        coarse_test_out=forward(trained_coarse_network,test_img);
        mid_img_feat=cat(3,test_img,coarse_test_out);
        final_test_out=forward(trained_fine_network,mid_img_feat);

        % conduct binarization (threshold of 0.5)
        thres=0.5;
        final_test_out=extractdata(final_test_out);
        binary_test_out=final_test_out>=thres;

        % compute accuracy for current image
        error=abs(binary_test_out-test_gt);
        error=sum(error,'all');
        total_error=total_error+error;
    end


    %% compute accuracy
    accuracy=1-(total_error/(row_size*col_size*num_test_data));
    msg=strcat("Current layer: ", lyr," , Accuracy of Pixel Prediction: ", num2str(accuracy));
    disp(msg);


end