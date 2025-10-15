function tortuosityImage = image_feature_en_face_generator_vessel_tortuosity(skeletonImage, vesselImage)
    %% Ensure input images are logical (binary)
    skeletonImage = logical(skeletonImage);
    vesselImage = logical(vesselImage);


    %% Label connected components in the skeleton
    % retrieve dimensional information and form a storage variable
    num_sample=size(vesselImage,4);
    num_layer=size(vesselImage,1);
    tortuosityImage = zeros(size(skeletonImage));

    
    %% loop through each layer in each sample and conduct vessel tortuosity calculation
    for i_sample=1:num_sample
        current_sample_image=skeletonImage(:,:,:,i_sample);
        current_sample_image=squeeze(current_sample_image);

        for i_layer=1:num_layer
            current_sample_layer_image=current_sample_image(i_layer,:,:);
            current_sample_layer_image=squeeze(current_sample_layer_image);
            CC = bwconncomp(current_sample_layer_image);
            num_vessels = CC.NumObjects;
            tortuosity_values = zeros(num_vessels, 1);
   

            %% Iterate through each vessel segment and conduct tortuosity calculation
            for i_vessel = 1:num_vessels
                vesselPixels = CC.PixelIdxList{i_vessel};
                [row, col] = ind2sub(size(current_sample_layer_image), vesselPixels);
        
                % Sort vessel points using a Minimum Spanning Tree approximation
                [sortedRow, sortedCol] = order_vessel_points(row, col);
        
                % Compute path length (sum of Euclidean distances between adjacent points)
                path_length = sum(sqrt(diff(sortedRow).^2 + diff(sortedCol).^2));

                % Compute Euclidean distance between endpoints
                if length(sortedRow) > 1
                    euclidean_distance = sqrt((sortedRow(end) - sortedRow(1))^2 + ...
                                      (sortedCol(end) - sortedCol(1))^2);
                else
                    euclidean_distance = 1; % Prevent division by zero
                end
        
                % Compute tortuosity
                tortuosity_values(i_vessel) = path_length / euclidean_distance;

                % Assign tortuosity value to vessel pixels
                tortuosityImage(vesselPixels) = tortuosity_values(i_vessel);
            end
        end
    end
    
end

function [orderedRow orderedCol] = order_vessel_points(row, col)
   
end

