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


function trained_network=training_function_en_face_seg(network,orig_data,orig_gt_data,oirg_valid_data,orig_valid_gt_data,num_epoch,learning_rate,lyr)

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
    valid_data=permute(oirg_valid_data,[1, 2, 5, 3, 4]);
    valid_gt_data=permute(orig_valid_gt_data,[1, 2, 5, 3, 4]);


    %% Initialize training progress
    % setup variable for storing training and validation loss
    train_loss=zeros(num_fold,num_epoch,1);
    valid_loss=zeros(num_valid_fold,num_epoch,1);
    % setup variable for update parameters
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.99;


    %% Training loop
    % load parameters if existed
    if exist("Phase_1_Segmentation/2D data function/models/temp","dir")
       % check if the directory is empty
       contents=dir("Phase_1_Segmentation/2D data function/models/temp");
       contents=contents(~ismember({contents.name},{'.','..'}));
       is_empty=isempty(contents);
       % if not empty load the file
       if ~is_empty
           load("Phase_1_Segmentation/2D data function/models/temp/net.mat");
           load("Phase_1_Segmentation/2D data function/models/temp/i_fold.mat");
           load("Phase_1_Segmentation/2D data function/models/temp/epoch.mat");
           start_epoch=epoch;
           start_fold=i_fold;
           net=net;
       % otherwise, initialize parameters
       else
           start_epoch=1;
           start_fold=1;
       end
    % if not, initialize the paramters
    else
        start_epoch=1;
        start_fold=1;
    end
 
    % loop through each fold
    num_epoch_plot=zeros(num_fold,1);
    net_storage=cell(num_fold);
    for i_fold=start_fold:num_fold
        % net=initialize(network);
        break_mark_epoch=false;
        termination_counter=0;

        % create fold data
        fold_data=data(:,:,:,:,i_fold);
        fold_gt_data=gt_data(:,:,:,:,i_fold);

        fold_data=dlarray(fold_data,"SSCB");
        % fold_data=dlarray(fold_data,"SSCB");
        fold_gt_data=dlarray(fold_gt_data,"SSCB");

        % initialize neural network
        net=initialize(network);

        % initialize weight for convolution (pca only)
        % net=initialize(net,fold_data);

        % loop through each epoch
        % loop through given number of epoches
        for epoch = start_epoch:num_epoch
            if break_mark_epoch
                break;
            end
    
            % initialize average gradient variable for the first epoch
            if epoch==1
                average_gradient=[];
                average_squared_gradient=[];
            end
    
            % use GPU if it can
            if canUseGPU
                data=gpuArray(data);
                gt_data=gpuArray(gt_data);
                valid_data=gpuArray(valid_data);
                valid_gt_data=gpuArray(valid_gt_data);
            end

            
            %% 2D Version loop 
            % create accumulated gradient variable (refresh in every epoch)
            accumulated_gradient=[];
            accumulated_loss=0;
    
            % loop through the training image one by one
            num_sample=size(data,5);

            for i_sample=1:num_sample
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
                    % if this is the last fold, update neural network
                    % Update parameters using Adam optimizer
                    [updated_net, average_gradient, average_squared_gradient] = adamupdate(net, accumulated_gradient,average_gradient,average_squared_gradient, epoch, learning_rate,gradientDecayFactor, squaredGradientDecayFactor);
                    % store training loss
                    t_loss=accumulated_loss/num_sample;
                    train_loss(i_fold,epoch,1)=t_loss;

                    % validation phase
                    fold_valid_data=valid_data(:,:,:,:,i_fold);
                    fold_valid_gt_data=valid_gt_data(:,:,:,:,i_fold);
                    fold_valid_data=dlarray(fold_valid_data,"SSCB");
                    fold_valid_gt_data=dlarray(fold_valid_gt_data,"SSCB");
                    v_loss=0;
                    for i_valid=1:num_valid_sample
                        batch_fold_valid_data=fold_valid_data(:,:,:,i_valid);
                        batch_gt_pred_valid=forward(net, batch_fold_valid_data);
                        batch_gt_valid=fold_valid_gt_data(:,:,:,i_valid);
                        batch_gt_pred_valid=squeeze(batch_gt_pred_valid);
                        batch_gt_valid=squeeze(batch_gt_valid);
                        smooth=1;
                        intersect=sum(batch_gt_pred_valid(:).*batch_gt_valid(:));
                        pred_sum=sum(batch_gt_pred_valid(:));
                        gt_sum=sum(batch_gt_valid(:));
                        dice_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
                        v_loss=v_loss+(1-dice_score);
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
                        net_storage{i_fold}=net;
                        start_epoch=1;
                        break;
                    end
                end

            end % end of sample

            % save temporarily intermediate file
            if ~exist("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp","dir")
                mkdir("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp")
            end
            save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/net.mat","net")
            save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/i_fold.mat","i_fold")
            save ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/epoch.mat","epoch")

        end % end of epoch
    
        % clean out unecessary variables
        clear loss gradients gt_pred_valid

        % delete intermediate files
        delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/i_fold.mat");
        delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/epoch.mat");

        % save current training and validation loss
        save ("~/data/klee232/train_en_face_seg_loss/rose_model_fine_64_aug_en_face_seg_dice_train_loss.mat", "train_loss")
        save ("~/data/klee232/valid_en_face_seg_loss/rose_model_fine_64_aug_en_face_seg_dice_valid_loss.mat", "valid_loss")
    end % end of fold

    % save the trained neural network and delete the intermediate file
    stored_data_path="~/data/klee232/trained_en_face_seg_model/";
    prototype_name="trained_rose_model_";
    layer_trained=lyr;
    augemented="_not_aug";
    loss_function="_dice.mat";
    complete_file_name=strcat(stored_data_path,prototype_name,layer_trained,augemented,loss_function);
    save(complete_file_name,'net_storage');
    delete ("~/GitHub/adp-3-human-retina/Phase_1_Segmentation/2D data function/models/temp/net.mat");


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
        ylabel("entropy loss");
        xlabel("training epoch");
    
        % save training loss plot
        if ~exist("Phase_1_Segmentation/2D data function/models/main_model", 'dir')
            mkdir("Phase_1_Segmentation/2D data function/models/main_model")
        end
        if ~exist("Phase_1_Segmentation/2D data function/models/training_loss_plot","dir")
            mkdir("Phase_1_Segmentation/2D data function/models/training_loss_plot")
        end
        stored_data_path="~/data/klee232/training_en_face_seg_loss_plot/";
        prototype_name="rose_model_fine_64_";
        layer_trained=lyr;
        augemented="_aug";
        loss_function="_dice_";
        fold_file=string(i_fold);
        type_file='.fig';
        full_file_name=strcat(stored_data_path,prototype_name,layer_trained,augemented,loss_function,fold_file,type_file);
        saveas(gcf, full_file_name);  % Save as a PNG image
    end
   

    %% return trained neural network
    trained_network=net_storage;
    

    toc
end


% Model gradients function
function [gradients, dice_loss] = modelGradients(network, data, gt_data)
    
    %% Forward Pass
    gt_pred = forward(network, data); 

    %% Backward Pass
    % calculate loss
    dice_loss=0;
    current_lyr_gt_pred=squeeze(gt_pred);
    current_lyr_gt_data=squeeze(gt_data);
    smooth=1;
    intersect=sum(current_lyr_gt_pred(:).*current_lyr_gt_data(:));
    pred_sum=sum(current_lyr_gt_pred(:));
    gt_sum=sum(current_lyr_gt_data(:));
    dice_score=(2*intersect+smooth)/(pred_sum+gt_sum+smooth);
    dice_loss=dice_loss+(1-dice_score);
    dice_loss=dlarray(dice_loss);

    % backpropagation
    gradients = dlgradient(dice_loss, network.Learnables, 'RetainData',false); 

end