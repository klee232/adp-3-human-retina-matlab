% Created by Kuan-Min Lee
% Created date: Mar. 6th, 2024
% All rights reserved to Leelab.ai

% Brief User Introduction:
% This function is built to train the implemented neural network

% Input Parameter:
% network: implemented deep learning neural netowrk (dlnetwork object)
% orig_data: data for training purpose (for our case, OCTA
% segmentation data)
% orig_gt_data: groundtruth for training purpose (for our case, labels for each sample)
% orig_valid_data: data for validation purpose (for our case, OCTA
% segmentation data for validation)
% orig_valid_gt_data: data for validation purpose (for our case, labels for
% each sample in validation data)
% num_epoch: number of epoches for training
% learning_rate: learning rate for optimization algorithm

% Output Parameter:
% trained_network: trained neural network (dlnetwork)


function trained_network=training_function_DVC_coarse(network, orig_data, orig_gt_data, orig_valid_data, orig_valid_gt_data,num_epoch,learning_rate,lyr)

    tic
    %% Convert data to dlarray
    % training dataset
    num_fold=size(orig_data,4);
    % input data
    data=permute(orig_data,[1, 2, 5, 3, 4]);    
    gt_data=permute(orig_gt_data,[1, 2, 5, 3, 4]);

    % validation data
    num_valid_fold=size(orig_valid_gt_data,4);
    num_valid_sample=size(orig_valid_gt_data,3);
    % input data
    valid_data=permute(orig_valid_data,[1, 2, 5, 3, 4]);
    valid_gt_data=permute(orig_valid_gt_data,[1, 2, 5, 3, 4]);


    %% Initialize training progress
    % setup variable for storing training and validation loss
    train_loss=zeros(num_fold,num_epoch,1);
    valid_loss=zeros(num_valid_fold,num_epoch,1);
    % setup variable for update parameters
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.99;


    %% Training loop
    %% load parameters if existed
    if exist("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp","dir")
       % check if the directory is empty
       contents=dir("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp");
       contents=contents(~ismember({contents.name},{'.','..'}));
       is_empty=isempty(contents);
       % if not empty load the file
       if ~is_empty
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/net.mat");
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_fold.mat");
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_sample.mat");
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/epoch.mat");
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/train_loss.mat");
           load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/valid_loss.mat");
           start_epoch=epoch;
           start_fold=i_fold;
           start_sample=i_sample;
           if start_sample~=1 && start_epoch~=1
              load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_loss.mat");
              load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_gradient.mat");
              load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_gradient.mat");
              load("Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_squared_gradient.mat");
           end
       % otherwise, initialize parameters
       else
           start_epoch=1;
           start_fold=1;
           start_sample=1;
           train_loss=zeros(num_fold,num_epoch,1);
           valid_loss=zeros(num_valid_fold,num_epoch,1);
           net_storage=cell(num_fold,1);
       end
    % if not, initialize the paramters
    else
        start_epoch=1;
        start_fold=1;
        start_sample=i_sample;
        net_storage=cell(num_fold,1);
    end
 

    %% loop through each fold
    num_epoch_plot=zeros(num_fold,1);
    for i_fold=start_fold:num_fold
        % net=initialize(network);
        break_mark_epoch=false;
        termination_counter=0;

        % create fold data
        fold_data=data(:,:,:,:,i_fold);
        fold_gt_data=gt_data(:,:,:,:,i_fold);
        fold_data=dlarray(fold_data,"SSCB");
        fold_gt_data=dlarray(fold_gt_data,"SSCB");

        % initialize neural network
        net=initialize(network);

        % loop through each epoch
        % loop through given number of epoches     
        for epoch = start_epoch:num_epoch
            if epoch==1
                average_gradient=[];
                average_squared_gradient=[];
            end
            if break_mark_epoch
                break;
            end
                
            % use GPU if it can
            if canUseGPU
                fold_data=gpuArray(fold_data);
                fold_gt_data=gpuArray(fold_gt_data);
                valid_data=gpuArray(valid_data);
                valid_gt_data=gpuArray(valid_gt_data);
            end

            
            %% 2D Version loop 
            % loop through the training image one by one
            num_sample=size(data,5);

            for i_sample=start_sample:num_sample
                if i_sample==1
                    accumulated_gradient=[];
                    accumulated_loss=0; 
                end

                % create batched data
                batch_fold_data=fold_data(:,:,:,i_sample);
                batch_fold_gt_data=fold_gt_data(:,:,:,i_sample);

                % Forward and backward pass             
                [gradients, loss]=dlfeval(@modelGradients, net, batch_fold_data, batch_fold_gt_data);

                % add on the current gradients and loss
                num_row=size(gradients,1);
                if isempty(accumulated_gradient)
                    accumulated_gradient=gradients;
                    for i_row=1:num_row
                        accumulated_gradient.Value{i_row,1}=(accumulated_gradient.Value{i_row,1})/num_sample;
                    end
                else
                    for i_row=1:num_row
                        accumulated_gradient.Value{i_row,1}= accumulated_gradient.Value{i_row,1}+(gradients.Value{i_row,1})/num_sample;
                    end
                end
                accumulated_loss=accumulated_loss+loss;

                % if the training reach the end of the sample do the following:
                % 1. Update network parameters
                % 2. store training loss
                % 3. conduct validation
                % 4. display accumulated loss in current epoch
                if i_sample==num_sample
                    % adding weight decay features
                    % weight_decay_factor=5e-4;
                    % learnables=net.Learnables;
                    % for i=1:size(accumulated_gradient,1)
                    %     w=learnables.Value{i};
                    %     g=accumulated_gradient.Value{i};
                    %     accumulated_gradient.Value{i}=g+weight_decay_factor*w;
                    % end
                    % if this is the last fold, update neural network
                    % Update parameters using Adam optimizer
                    [updated_net, average_gradient, average_squared_gradient] = adamupdate(net, accumulated_gradient,average_gradient,average_squared_gradient, epoch, learning_rate,gradientDecayFactor, squaredGradientDecayFactor);
                    % conduct validation
                    % store the training loss
                    t_loss=accumulated_loss/num_sample;
                    train_loss(i_fold,epoch,1)=t_loss;

                    % validation phase
                    fold_valid_data=valid_data(:,:,:,:,i_fold);
                    fold_valid_gt_data=valid_gt_data(:,:,:,:,i_fold);
                    fold_valid_data=dlarray(fold_valid_data,"SSCB");
                    fold_valid_gt_data=dlarray(fold_valid_gt_data,"SSCB");
                    v_loss=0;
                    for i_valid=1:num_valid_sample
                        % forward input
                        batch_fold_valid_data=fold_valid_data(:,:,:,i_valid);
                        batch_fold_gt_pred_valid=forward(updated_net, batch_fold_valid_data);

                        % grab out groundtruth
                        batch_fold_gt_valid=fold_valid_gt_data(:,:,:,i_valid);

                        % calculate the loss
                        batch_fold_gt_pred_valid=squeeze(batch_fold_gt_pred_valid);
                        batch_fold_gt_valid=squeeze(batch_fold_gt_valid);
                        % mse loss
                        mse_loss=mean((batch_fold_gt_pred_valid(:)-batch_fold_gt_valid(:)).^2);

                        % dice loss
                        % dice_loss=0;
                        % smooth=1e-7;
                        % intersect=sum(batch_fold_gt_pred_valid(:).*batch_fold_gt_valid(:));
                        % pred_sum=sum(batch_fold_gt_pred_valid(:));
                        % gt_sum=sum(batch_fold_gt_valid(:));
                        % dice_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
                        % dice_loss=dice_loss+(1-dice_score);

                        % dice centerline loss
                        % dice_ct_loss=0;
                        % batch_fold_gt_pred_valid_bin=imbinarize(extractdata(batch_fold_gt_pred_valid));
                        % batch_fold_gt_valid_bin=imbinarize(extractdata(batch_fold_gt_valid));
                        % batch_fold_gt_pred_valid_ct=bwskel(batch_fold_gt_pred_valid_bin);
                        % batch_fold_gt_valid_ct=bwskel(batch_fold_gt_valid_bin);
                        % smooth=1e-7;
                        % intersect=sum(batch_fold_gt_pred_valid_ct(:).*batch_fold_gt_valid_ct(:));
                        % pred_sum=sum(batch_fold_gt_pred_valid_ct(:));
                        % gt_sum=sum(batch_fold_gt_valid_ct(:));
                        % dice_ct_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
                        % dice_ct_loss=dice_ct_loss+(1-dice_ct_score);

                        % tversky loss
                        epsilon=1e-6;
                        alpha=0.3;
                        beta=0.7;
                        TP=sum(batch_fold_gt_pred_valid(:).*batch_fold_gt_valid(:));
                        FP=sum(batch_fold_gt_pred_valid(:).*(1-batch_fold_gt_valid(:)));
                        FN=sum((1-batch_fold_gt_pred_valid(:)).*batch_fold_gt_valid(:));
                        tversky_loss=1-TP/(TP+alpha*FP+beta*FN+epsilon);

                        batch_fold_total_loss=mse_loss+tversky_loss;

                        v_loss=v_loss+batch_fold_total_loss;
                    end
                    v_loss=v_loss/(num_valid_sample);
                    valid_loss(i_fold,epoch,1)=v_loss;

                    % Display progress
                    if mod(epoch, 10) == 0
                        msg=strcat("Current fold: ", string(i_fold)," Epoch: ", string(epoch), ", training Loss: ", string(t_loss));
                        disp(msg);
                        msg=strcat("Current fold: ", string(i_fold)," Epoch: ", string(epoch), ", validation Loss: ", string(v_loss));
                        disp(msg);
                    end

                    % check overfitting condition
                    if epoch>1
                        % update network only if current validation loss is smaller
                        % calculate average validation loss for current
                        % epoch
                        if ~exist('best_valid_loss','var')
                            best_valid_loss=valid_loss(i_fold,epoch);
                        end
                        if v_loss<best_valid_loss
                            best_valid_loss=v_loss;
                            net=updated_net;
                        else
                            termination_counter=termination_counter+1;
                        end
                    else
                        best_valid_loss=v_loss;
                        net=updated_net;
                    end

                    clear v_loss

                    % if the termination_counter hits 10 jumps out the
                    % training
                    if termination_counter==50
                        break_mark_epoch=true;
                        num_epoch_plot(i_fold,1)=epoch;
                        net_storage{i_fold,1}=net;
                        start_epoch=1;
                        break;
                    end
                end

                 % save temporarily intermediate file
                if ~exist("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp","dir")
                    mkdir("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp")
                end

                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_loss.mat","accumulated_loss","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_gradient.mat","accumulated_gradient","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_gradient.mat","average_gradient","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_squared_gradient.mat","average_squared_gradient","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/net.mat","net_storage","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_fold.mat","i_fold","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_sample.mat","i_sample","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/epoch.mat","epoch","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/train_loss.mat","train_loss","-v7.3")
                save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/valid_loss.mat","valid_loss","-v7.3")

            end % end of sample
           
        end % end of epoch
    
        % store the trained outcome
        net_storage{i_fold,1}=net;
        num_epoch_plot(i_fold,1)=epoch;

        % clean out unecessary variables
        clear loss gradients gt_pred_valid
       
        % save current training and validation loss
        save ("~/data/klee232/train_en_face_seg_loss/coarse_stage_model_9_DVC_triaug_msePlustversky_batch2_train_loss.mat", "train_loss")
        save ("~/data/klee232/valid_en_face_seg_loss/coarse_stage_model_9_DVC_triaug_msePlustversky_batch2_valid_loss.mat", "valid_loss")
    end % end of fold

    % delete intermediate files
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/net.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_fold.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/epoch.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/i_sample.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/train_loss.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/valid_loss.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_loss.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/accumulated_gradient.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_gradient.mat");
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/training_and_testing_functions/DVC_coarse_temp/average_squared_gradient.mat");

    % save the trained neural network and delete the intermediate file
    stored_data_path="~/data/klee232/trained_en_face_seg_model/";
    prototype_name="trained_coarse_stage_model_9_";
    layer_trained=lyr;
    augemented="_triaug";
    loss_function="_msePlustversky.mat";
    complete_file_name=strcat(stored_data_path,prototype_name,layer_trained,augemented,loss_function);
    save(complete_file_name,'net_storage',"-v7.3");


    %% generate training plot
    for i_fold=1:num_fold
        figure;
        plot(1:num_epoch_plot(i_fold,1),train_loss(i_fold,1:num_epoch_plot(i_fold,1),1),1:num_epoch_plot(i_fold,1),valid_loss(i_fold,1:num_epoch_plot(i_fold,1),1));
        grid on
        legend('training loss','validation loss');
        img_title="training loss plot for fold ";
        fold_num=string(i_fold);
        img_full_title=strcat(img_title,fold_num);
        title(img_full_title);
        ylabel("mse + tversky loss");
        xlabel("training epoch");
    
        % save training and validation loss plot
        stored_data_path="~/data/klee232/training_en_face_seg_loss_plot/";
        prototype_name="coarse_stage_model_9_";
        layer_trained=lyr;
        augemented="_triaug";
        loss_function="_msePlustversky_";
        fold_file=string(i_fold);
        type_file='.fig';
        full_file_name=strcat(stored_data_path,prototype_name,layer_trained,augemented,loss_function,fold_file,type_file);
        saveas(gcf, full_file_name);  % Save as a fig image
    end
   

    %% return trained neural network
    trained_network=net_storage;
    

    toc
end


% Model gradients function
function [gradients, total_loss] = modelGradients(network, data, gt_data)
    
    %% Forward Pass
    gt_pred = forward(network, data); 

    %% Backward Pass
    % calculate loss
    % mse loss
    current_lyr_gt_pred=squeeze(gt_pred);
    current_lyr_gt_data=squeeze(gt_data);
    mse_loss=mean((current_lyr_gt_data(:)-current_lyr_gt_pred(:)).^2);
    mse_loss=dlarray(mse_loss);

    % dice loss
    % dice_loss=0;
    % current_lyr_gt_pred=squeeze(gt_pred);
    % current_lyr_gt_data=squeeze(gt_data);
    % smooth=1e-7;
    % intersect=sum(current_lyr_gt_pred(:).*current_lyr_gt_data(:));
    % pred_sum=sum(current_lyr_gt_pred(:));
    % gt_sum=sum(current_lyr_gt_data(:));
    % dice_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
    % dice_loss=dice_loss+(1-dice_score);

    % dice centerline loss
    % dice_ct_loss=0;
    % current_lyr_gt_pred_bin=imbinarize(extractdata(current_lyr_gt_pred));
    % current_lyr_gt_data_bin=imbinarize(extractdata(current_lyr_gt_data));
    % current_lyr_gt_ct_pred=bwskel(current_lyr_gt_pred_bin);
    % current_lyr_gt_ct_data=bwskel(current_lyr_gt_data_bin);
    % smooth=1e-7;
    % intersect=sum(current_lyr_gt_ct_pred(:).*current_lyr_gt_ct_data(:));
    % pred_sum=sum(current_lyr_gt_ct_pred(:));
    % gt_sum=sum(current_lyr_gt_ct_data(:));
    % dice_ct_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
    % dice_ct_loss=dice_ct_loss+(1-dice_ct_score);

    % tversky loss
    epsilon=1e-6;
    alpha=0.3;
    beta=0.7;
    TP=sum(current_lyr_gt_pred(:).*current_lyr_gt_data(:));
    FP=sum(current_lyr_gt_pred(:).*(1-current_lyr_gt_data(:)));
    FN=sum((1-current_lyr_gt_pred(:)).*current_lyr_gt_data(:));
    tversky_loss=1-TP/(TP+alpha*FP+beta*FN+epsilon);

    total_loss=mse_loss+tversky_loss;

    % backpropagation
    gradients = dlgradient(total_loss, network.Learnables, 'RetainData',false); 

end