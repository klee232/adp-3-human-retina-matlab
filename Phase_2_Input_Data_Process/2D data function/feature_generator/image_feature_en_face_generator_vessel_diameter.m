function vessel_diameter_image = image_feature_en_face_generator_vessel_diameter(skeleton, vessel_mask)
    % Ensure images are logical (binary)
    skeleton = logical(skeleton);
    vessel_mask = logical(vessel_mask);
    
    % Get vessel skeleton points
    num_layer=size(skeleton,1);
    num_sample=size(skeleton,4);
    % Initialize vessel diameter storage
    vessel_diameter_skeleton_image = zeros(size(skeleton));

    for i_sample=1:num_sample
        for i_lyr=1:num_layer
            current_skeleton=skeleton(i_lyr,:,:,i_sample);
            current_skeleton=squeeze(current_skeleton);
            current_vessel_mask=vessel_mask(i_lyr,:,:,i_sample);
            current_vessel_mask=squeeze(current_vessel_mask);
            [skeleton_x, skeleton_y] = find(current_skeleton);
            numPoints = length(skeleton_x);
    
            % Process each skeleton point
            for i = 1:numPoints
                % Current skeleton point
                x = skeleton_x(i);
                y = skeleton_y(i);
        
                % Compute perpendicular profile
                [diameter] = measure_vessel_width(x, y, current_vessel_mask);
                
                % Assign diameter to skeleton pixel
                vessel_diameter_skeleton_image(i_lyr, x, y, i_sample) = diameter;
            end
        end
    end

    % Propagate skeleton diameters back to the full vessel mask
    vessel_diameter_image = propagate_diameter_to_mask(vessel_diameter_skeleton_image, vessel_mask);
    
    
end

function [diameter] = measure_vessel_width(x, y, vessel_mask)
    % Define search directions (horizontal, vertical, diagonal) (i think
    % this is wrong) (this shouldn't be all directions. this should be
    % orthongonal direction)
    directions = [1 0; -1 0; 0 1; 0 -1; 1 1; -1 -1; 1 -1; -1 1];
    
    max_distance = 0;
    
    % Search in all directions
    for d = 1:size(directions,1)
        dir_x = directions(d,1);
        dir_y = directions(d,2);
        
        % Move outward until hitting vessel boundary
        dist = 0;
        while true
            check_x = x + dist * dir_x;
            check_y = y + dist * dir_y;
            
            % Stop if outside vessel mask or image bounds
            if check_x < 1 || check_y < 1 || check_x > size(vessel_mask,1) || check_y > size(vessel_mask,2)
                break;
            end
            if ~vessel_mask(round(check_x), round(check_y)) % If outside vessel
                break;
            end
            
            dist = dist + 1;
        end
        
        % Update max diameter
        if dist > max_distance
            max_distance = dist;
        end
    end
    
    % Vessel diameter is twice the maximum distance from center
    diameter = 2 * max_distance;
end

function projected_diameter = propagate_diameter_to_mask(skeleton_diameter_map, vessel_mask)
    % Initialize projected diameter map
    projected_diameter = zeros(size(vessel_mask));

    % Find vessel pixels
    num_sample=size(skeleton_diameter_map,4);
    num_layer=size(skeleton_diameter_map,1);

    for i_sample=1:num_sample
        for i_lyr=1:num_layer
            current_skeleton_diameter_map=skeleton_diameter_map(i_lyr,:,:,i_sample);
            current_skeleton_diameter_map=squeeze(current_skeleton_diameter_map);
            current_vessel_mask=vessel_mask(i_lyr,:,:,i_sample);
            current_vessel_mask=squeeze(current_vessel_mask);   
            [vessel_x, vessel_y] = find(current_vessel_mask);
            [skeleton_x, skeleton_y, skeleton_diameters] = find(current_skeleton_diameter_map);

            % Assign the nearest skeleton pixel's diameter to each vessel pixel
            for i = 1:length(vessel_x)
                x = vessel_x(i);
                y = vessel_y(i);
        
                % Find the closest skeleton point
                distances = sqrt((skeleton_x - x).^2 + (skeleton_y - y).^2);
                [~, minIdx] = min(distances);
        
                % Assign nearest skeleton's diameter value
                projected_diameter(i_lyr, x, y, i_sample) = skeleton_diameters(minIdx);
            end
        end
    end
end