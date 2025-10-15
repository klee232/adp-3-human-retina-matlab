% Created by Kuan-Min Lee
% Created date: Dec. 13th, 2024
% All rights reserved to Leelab.ai

% Brief User Introducttion:
% The following codes are constructed to compute confusion matrix for
% trained neural network

% Inputs Parameters:
% pred_gt_data - prediction output from trained neural network
% gt_data - groundtruth from dataset

% Output Parameters:
% confusion_matrix - generated confusion matrix


function confusion_matrix=compute_Confusion_Matrix(pred_gt_data,gt_data)
    
    %% setup confusion matrix computation
    % fold 1
    gt_data_f1=gt_data(:,:,1);
    pred_gt_data_f1=pred_gt_data(:,:,1);
    confusion_matrix_f1=zeros(size(gt_data_f1,1),size(gt_data_f1,1));
    num_sample=size(gt_data_f1,2);
    for i_sample=1:num_sample
        % grab out current sample
        current_data=gt_data_f1(:,i_sample);
        current_pred_data=pred_gt_data_f1(:,i_sample);
        [~,current_gt_class]=max(current_data);
        [~,current_pred_class]=max(current_pred_data);
        confusion_matrix_f1(current_gt_class,current_pred_class)=confusion_matrix_f1(current_gt_class,current_pred_class)+1;
    end
    figure;
    imagesc(confusion_matrix_f1);
    colorbar;
    colormap('jet');
    xlabel('prediction label');
    ylabel('actual label');
    title('confusion matrix fold 1');
    saveas(gcf, 'Phase_3_Prediction_model/models/testing_plot/train_original_network_mean_removal_pca_confusion.png');  % Save as a PNG image

    % % fold 2
    % gt_data_f2=gt_data(:,:,1);
    % pred_gt_data_f2=pred_gt_data(:,:,2);
    % confusion_matrix_f2=zeros(size(gt_data_f2,1),size(gt_data_f2,1));
    % num_sample=size(gt_data_f2,2);
    % for i_sample=1:num_sample
    %     % grab out current sample
    %     current_data=gt_data_f2(:,i_sample);
    %     current_pred_data=pred_gt_data_f2(:,i_sample);
    %     [~,current_gt_class]=max(current_data);
    %     [~,current_pred_class]=max(current_pred_data);
    %     confusion_matrix_f2(current_gt_class,current_pred_class)=confusion_matrix_f2(current_gt_class,current_pred_class)+1;
    % end
    % figure;
    % imagesc(confusion_matrix_f2);
    % colorbar;
    % colormap('jet');
    % xlabel('prediction label');
    % ylabel('actual label');
    % title('confusion matrix fold 2');
    % saveas(gcf, 'Phase_3_Prediction_model/models/testing_plot/train_original_network_mean_removal_pca_50_confusion_f2.png');  % Save as a PNG image
    % 
    % % fold 3
    % gt_data_f3=gt_data(:,:,1);
    % pred_gt_data_f3=pred_gt_data(:,:,3);
    % confusion_matrix_f3=zeros(size(gt_data_f3,1),size(gt_data_f3,1));
    % num_sample=size(gt_data_f3,2);
    % for i_sample=1:num_sample
    %     % grab out current sample
    %     current_data=gt_data_f3(:,i_sample);
    %     current_pred_data=pred_gt_data_f3(:,i_sample);
    %     [~,current_gt_class]=max(current_data);
    %     [~,current_pred_class]=max(current_pred_data);
    %     confusion_matrix_f3(current_gt_class,current_pred_class)=confusion_matrix_f3(current_gt_class,current_pred_class)+1;
    % end
    % figure;
    % imagesc(confusion_matrix_f3);
    % colorbar;
    % colormap('jet');
    % xlabel('prediction label');
    % ylabel('actual label');
    % title('confusion matrix fold 3');
    % saveas(gcf, 'Phase_3_Prediction_model/models/testing_plot/train_original_network_mean_removal_pca_50_confusion_f3.png');  % Save as a PNG image
    % 
    % % fold 4
    % gt_data_f4=gt_data(:,:,1);
    % pred_gt_data_f4=pred_gt_data(:,:,4);
    % confusion_matrix_f4=zeros(size(gt_data_f4,1),size(gt_data_f4,1));
    % num_sample=size(gt_data_f4,2);
    % for i_sample=1:num_sample
    %     % grab out current sample
    %     current_data=gt_data_f4(:,i_sample);
    %     current_pred_data=pred_gt_data_f4(:,i_sample);
    %     [~,current_gt_class]=max(current_data);
    %     [~,current_pred_class]=max(current_pred_data);
    %     confusion_matrix_f4(current_gt_class,current_pred_class)=confusion_matrix_f4(current_gt_class,current_pred_class)+1;
    % end
    % figure;
    % imagesc(confusion_matrix_f4);
    % colorbar;
    % colormap('jet');
    % xlabel('prediction label');
    % ylabel('actual label');
    % title('confusion matrix fold 4');
    % saveas(gcf, 'Phase_3_Prediction_model/models/testing_plot/train_original_network_mean_removal_pca_50_confusion_f4.png');  % Save as a PNG image
    % 

    % confusion_matrix=cat(3,confusion_matrix_f1,confusion_matrix_f2,confusion_matrix_f3,confusion_matrix_f4);
    confusion_matrix=confusion_matrix_f1;

end