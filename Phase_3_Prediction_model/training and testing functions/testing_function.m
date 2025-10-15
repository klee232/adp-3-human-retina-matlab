% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is built to test the implemented neural network

% Input Parameter:
% network: trained deep learning neural netowrk (dlnetwork object)
% orig_test_data: data for testing purpose (for our case, OCTA
% segmentation data)
% orig_test_gt_data: groundtruth for testing purpose (for our case, labels for each sample)

% Output Parameter:
% test_loss: testing loss (testing entropy loss)

function [test_loss,confusion_matrix]=testing_function(network,orig_test_data,orig_test_gt_data,num_fold)
    
    %% convert the data to dlarray
    test_data=permute(orig_test_data,[1, 2, 3, 6, 4, 5]);
    num_sample=size(orig_test_gt_data,1);
    test_classes=unique(orig_test_gt_data);
    num_test_classes=size(test_classes,1);
    test_gt_data=zeros(num_test_classes,num_sample,1);
    for i_sample=1:num_sample
        current_data=orig_test_gt_data(i_sample,1);
        [ind]=find(test_classes==current_data);
        class_vector=zeros(1,num_test_classes);
        class_vector(1,ind)=1;
        test_gt_data(:,i_sample,1)=class_vector;
    end
    test_gt_data=dlarray(test_gt_data);


    %% calculate cross entropy loss 
    num_sample=size(orig_test_gt_data,1);
    fold_loss=zeros(num_fold);
    out_storage=zeros(num_test_classes,num_sample,num_fold);
    for i_fold=1:num_fold
        accumulated_loss=0;
        for i_sample=1:num_sample
            % grab out fold neural network
            rng(0); % Set random seed
            net=initialize(network{i_fold});

            % create batched data
            batch_data=test_data(:,:,:,:,i_sample,1);
            batch_gt_data=test_gt_data(:,i_sample,1);
            batch_data=dlarray(batch_data,"SSSC");
            batch_gt_data=dlarray(batch_gt_data);

            % setup storage variable for storing test loss
            gt_pred_test=predict(net,batch_data);
            current_loss=crossentropy(gt_pred_test,batch_gt_data);
    
            % add on the current loss
            accumulated_loss=accumulated_loss+current_loss/(num_sample);
    
            % store model outcome
            out_storage(:,i_sample,i_fold)=gt_pred_test;
        end

        % store fold loss
        fold_loss(i_fold)=accumulated_loss;
    end

    % return testing loss
    test_loss=fold_loss;

    %% ROC curve production

    %% Fold 1
    % compute HR ROC
    [HR_FPR, HR_TPR, HR_AUC]=compute_ROC(test_gt_data, out_storage, 1,'HR');
    % compute LR ROC
    % [LR_FPR, LR_TPR, LR_AUC]=compute_ROC(test_gt_data, out_storage, 1,'LR');
    % compute MCI ROC
    [MCI_FPR, MCI_TPR, MCI_AUC]=compute_ROC(test_gt_data, out_storage, 1,'MCI');



    %% Confusion matrix production

    confusion_matrix=compute_Confusion_Matrix(out_storage,test_gt_data);
    

end
