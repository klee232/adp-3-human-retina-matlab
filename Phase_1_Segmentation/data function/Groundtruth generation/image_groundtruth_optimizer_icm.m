% Created by Kuan-Min Lee
% Createed date: Sep. 23rd, 2024

% Brief User Introduction:
% This function is built to optimize the binary groundtruth for the training
% 3D OCTA image network. 

% Input parameter:
% orig_img: input original OCTA image (3D image array)
% initial_gt: input OCTA groundtruth image generated from previous phase
% (3D image array)
% smoothness_weight: penalization coefficients applied for i and j are
% difference classes (default setting as 0.5)
% max_iterations: maximum number of iteration (default setting as 500)


% Output:
% optimized_labels: optimized groundtruth (3D image array)


function [optimized_labels,picture_obj_octa] = image_groundtruth_optimizer_icm(orig_image, initial_gt, seg_mask,smoothness_weight, max_iterations,picture_obj_octa)
    %% create corresponding binary mask for three parts
    % (surface, deep, choroid)
    mask_orig=seg_mask;
    mask_surf=zeros(size(mask_orig));
    mask_deep=zeros(size(mask_orig));
    mask_chor=zeros(size(mask_orig));
    mask_surf(mask_orig>=2 & mask_orig<=6)=1;
    mask_deep(mask_orig==8)=1;
    mask_chor(mask_orig==14)=1;

    %% partition the original image into three parts (surface, deep and choroid layers)
    orig_image_surf=orig_image.*double(mask_surf);
    orig_image_deep=orig_image.*double(mask_deep);
    orig_image_chor=orig_image.*double(mask_chor);
    initial_gt_surf=initial_gt.*double(mask_surf);
    initial_gt_deep=initial_gt.*double(mask_deep);
    initial_gt_chor=initial_gt.*double(mask_chor);

    %% calculate the mean pixel value for each label in each layer
    % separate three layers
    % (1: vessel, 0: background)
    surf_ind=max(mask_surf,[],[2 3]);
    deep_ind=max(mask_deep,[],[2 3]);
    chor_ind=max(mask_chor,[],[2 3]);
    limit_orig_image_surf=orig_image_surf(surf_ind==1,:,:);
    limit_orig_image_deep=orig_image_deep(deep_ind==1,:,:);
    limit_orig_image_chor=orig_image_chor(chor_ind==1,:,:);
    limit_initial_gt_surf=initial_gt_surf(surf_ind==1,:,:);
    limit_initial_gt_deep=initial_gt_deep(deep_ind==1,:,:);
    limit_initial_gt_chor=initial_gt_chor(chor_ind==1,:,:);

    limit_orig_image_surf_ves=limit_orig_image_surf(limit_initial_gt_surf==1);
    limit_orig_image_surf_back=limit_orig_image_surf(limit_initial_gt_surf==0);
    limit_orig_image_deep_ves=limit_orig_image_deep(limit_initial_gt_deep==1);
    limit_orig_image_deep_back=limit_orig_image_deep(limit_initial_gt_deep==0);
    limit_orig_image_chor_ves=limit_orig_image_chor(limit_initial_gt_chor==1);
    limit_orig_image_chor_back=limit_orig_image_chor(limit_initial_gt_chor==0);

    % calculate the the mean pixels from three layers
    mean_surf_ves=mean(limit_orig_image_surf_ves,"all");
    mean_surf_back=mean(limit_orig_image_surf_back,"all");
    mean_deep_ves=mean(limit_orig_image_deep_ves,"all");
    mean_deep_back=mean(limit_orig_image_deep_back,"all");
    mean_chor_ves=mean(limit_orig_image_chor_ves,"all");
    mean_chor_back=mean(limit_orig_image_chor_back,"all");

    %% set up stopping threshold for each layer
    [img_depth_surf,img_height_surf,img_width_surf]=size(limit_orig_image_surf);
    [img_depth_deep,img_height_deep,img_width_deep]=size(limit_orig_image_deep);
    [img_depth_chor,img_height_chor,img_width_chor]=size(limit_orig_image_chor);
    stopping_threshold_surf=0.001*(img_depth_surf*img_height_surf*img_width_surf);
    stopping_threshold_deep=0.001*(img_depth_deep*img_height_deep*img_width_deep);
    stopping_threshold_chor=0.001*(img_depth_chor*img_height_chor*img_width_chor);

    %% conduct optimization for each layer
    [limit_initial_gt_surf,num_changes_store_surf]=optimizer_icm(limit_orig_image_surf,limit_initial_gt_surf,mean_surf_ves,mean_surf_back,img_depth_surf,img_height_surf,img_width_surf,stopping_threshold_surf,smoothness_weight,max_iterations);
    [limit_initial_gt_deep,num_changes_store_deep]=optimizer_icm(limit_orig_image_deep,limit_initial_gt_deep,mean_deep_ves,mean_deep_back,img_depth_deep,img_height_deep,img_width_deep,stopping_threshold_deep,smoothness_weight,max_iterations);
    [limit_initial_gt_chor,num_changes_store_chor]=optimizer_icm(limit_orig_image_chor,limit_initial_gt_chor,mean_chor_ves,mean_chor_back,img_depth_chor,img_height_chor,img_width_chor,stopping_threshold_chor,smoothness_weight,max_iterations);
    
    %% visualize the number of pixel changed in every iteration 
    figure;
    plot(1:length(num_changes_store_surf),num_changes_store_surf);
    figure;
    plot(1:length(num_changes_store_deep),num_changes_store_deep);
    figure;
    plot(1:length(num_changes_store_chor),num_changes_store_chor);

    %% store optimized information
    labels=zeros(size(initial_gt));
    labels(surf_ind==1,:,:)=labels(surf_ind==1,:,:)+limit_initial_gt_surf;
    labels(deep_ind==1,:,:)=labels(deep_ind==1,:,:)+limit_initial_gt_deep;
    labels(chor_ind==1,:,:)=labels(chor_ind==1,:,:)+limit_initial_gt_chor;
    labels(labels>1)=1;
    optimized_labels=labels;
    picture_obj_octa.binary_image_opt=optimized_labels;

end

% optimizer icm function
function [labels,num_changes_store]=optimizer_icm(orig_image,labels,mean_ves,mean_back,img_depth,img_height,img_width,stopping_threshold,smoothness_weight,max_iterations)
    % set up variable for storage
    num_changes_store=zeros(max_iterations,1);

    % setup initial labels
    orig_label=labels;

    % ICM optimization for MGRF
    for i_iter = 1:max_iterations
        % Iterate over each pixel
        for i_depth=1:img_depth
            for i_height=1:img_height
                for i_width=1:img_width
                    % Unary potentials (likelihood based on intensity)
                    unary_0=unary_potential(0, orig_image(i_depth,i_height,i_width), mean_ves, mean_back);
                    unary_1=unary_potential(1, orig_image(i_depth,i_height,i_width), mean_ves, mean_back);
    
                    % Pairwise potential (6-connected neighbors)
                    % create neighborhood indices
                    pairwise_0 = 0;
                    pairwise_1 = 0;
                    neighbors = [i_depth-1 i_height i_width; ...
                                 i_depth+1 i_height i_width; ...
                                 i_depth i_height-1 i_width; ...
                                 i_depth i_height+1 i_width; ...
                                 i_depth i_height i_width-1; ...
                                 i_depth i_height i_width+1]; % 6-connected neighborhood
                    % loop through each neighbor pixel
                    for i_neighbor=1:size(neighbors, 1)
                        % grab out current neighbor index
                        neighbor_depth=neighbors(i_neighbor,1);
                        neighbor_height=neighbors(i_neighbor,2);
                        neighbor_width=neighbors(i_neighbor,3);
                        % check if the current neighbor index is within the
                        % boundary
                        % calculate the probability potential only if the
                        % location is within the boundary
                        if (neighbor_depth>=1 && neighbor_depth<=img_depth) && ...
                           (neighbor_height>=1 && neighbor_height<=img_height) && ...
                           (neighbor_width>=1 && neighbor_width<=img_width)
                            neighbor_label=labels(neighbor_depth,neighbor_height,neighbor_width);
                            % calculate pairwise potential 
                            pairwise_0=pairwise_0+pairwise_potential(0, neighbor_label, smoothness_weight);
                            pairwise_1=pairwise_1+pairwise_potential(1, neighbor_label, smoothness_weight);
                        end
                    end
    
                    % Total energy for each label
                    energy_0=unary_0+pairwise_0;
                    energy_1=unary_1+pairwise_1;
    
                    % Choose the label with the minimum energy
                    if energy_0<energy_1
                        labels(i_depth,i_height,i_width)=0;
                    else
                        labels(i_depth,i_height,i_width)=1;
                    end

                end % end for i_width
            end % end for i_height
        end % end for i_depth

        fprintf('Iteration %d completed.\n', i_iter);
        
        % Check for convergence (based on label changes)
        if i_iter>=2
            num_changes = sum(labels(:)~=previous_labels(:));
            % store the number of changes in this iteration
            num_changes_store(i_iter,1)=num_changes;
            if num_changes<stopping_threshold
                break;  % Stop if the number of label changes is below threshold
            end
        else
            num_changes = sum(labels(:)~=orig_label(:));
            % store the number of changes in this iteration
            num_changes_store(i_iter,1)=num_changes;
            if num_changes<stopping_threshold
                break;  % Stop if the number of label changes is below threshold
            end
        end

        % Update for the next iteration
        previous_labels = labels;  

    end % end for i_iter    

end


% Unary potential function
function potential = unary_potential(label, pixel_intensity, fg_mean, bg_mean)
    % Calculate unary potential based on pixel intensity and label
    if label == 1
        potential=(pixel_intensity-fg_mean)^2; % Foreground likelihood
    else
        potential=(pixel_intensity-bg_mean)^2; % Background likelihood
    end
end

% Pairwise potential function
function potential = pairwise_potential(label1,label2,smoothness_weight)
    % Penalize different labels for neighboring pixels
    if label1 ~= label2
        potential=smoothness_weight;
    else
        potential=0;
    end
end

