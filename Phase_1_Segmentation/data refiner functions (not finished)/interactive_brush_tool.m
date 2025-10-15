function interactive_brush_tool()
    % Create a sample image (a grayscale image)
    img = zeros(100, 100);
    
    % Create a figure window and display the image
    hFig = figure;
    hAx = axes('Parent', hFig);
    hImg = imshow(img, 'Parent', hAx);
    
    % Set up the brush tool
    brushSize = 5; % Size of the brush
    set(hFig, 'WindowButtonDownFcn', @startBrush);
    set(hFig, 'WindowButtonUpFcn', @stopBrush);
    set(hFig, 'WindowButtonMotionFcn', @brush);

    function startBrush(~, ~)
        % This function starts the brushing action
        brush();
    end

    function stopBrush(~, ~)
        % This function stops the brushing action
        set(hFig, 'WindowButtonMotionFcn', @brush);
    end

    function brush(~, ~)
        % This function changes the pixel values to 1 where the user clicks
        C = get(hAx, 'CurrentPoint');
        x = round(C(1,1));
        y = round(C(1,2));
        
        % Ensure coordinates are within image boundaries
        if x > 0 && y > 0 && x <= size(img, 2) && y <= size(img, 1)
            % Define the region around the clicked point
            xRange = max(1, x-brushSize):min(size(img, 2), x+brushSize);
            yRange = max(1, y-brushSize):min(size(img, 1), y+brushSize);
            
            % Set the selected region to 1
            img(yRange, xRange) = 1;
            
            % Update the displayed image
            set(hImg, 'CData', img);
        end
    end
end