function interactive_image_brush(image)
    % Check if an image is provided, otherwise use a sample image
    if nargin < 1
        image = imread('peppers.png'); % Load a default image if none is provided
    end

    % Create a figure with multiple subplots
    hFig = figure;
    hAx1 = subplot(2,2,1); imshow(image, 'Parent', hAx1); title('Subplot 1');
    hAx2 = subplot(2,2,2); imshow(image, 'Parent', hAx2); title('Subplot 2');
    hAx3 = subplot(2,2,3); imshow(image, 'Parent', hAx3); title('Subplot 3');
    hAx4 = subplot(2,2,4); imshow(image, 'Parent', hAx4); title('Subplot 4');
    
    % Choose the axis where painting is allowed
    targetAxis = hAx1; % Change this to hAx2, hAx3, or hAx4 as needed
    hold(targetAxis, 'on');
    
    % Set up callback functions
    set(hFig, 'WindowButtonDownFcn', @startPainting);
    set(hFig, 'KeyPressFcn', @keyPressHandler);
    
    % Initialize painting parameters
    brushSize = 5;  % Size of the brush
    brushColor = [1, 0, 0]; % Color of the brush (red)
    
    function startPainting(~, ~)
        % Get the position of the mouse click
        mousePos = get(targetAxis, 'CurrentPoint');
        x = round(mousePos(1, 1));
        y = round(mousePos(1, 2));
        
        if isInsideAxis(x, y, targetAxis)
            paint(x, y);
            set(hFig, 'WindowButtonMotionFcn', @continuePainting);
            set(hFig, 'WindowButtonUpFcn', @stopPainting);
        end
    end
    
    function continuePainting(~, ~)
        % Continue painting as the mouse moves
        mousePos = get(targetAxis, 'CurrentPoint');
        x = round(mousePos(1, 1));
        y = round(mousePos(1, 2));
        
        if isInsideAxis(x, y, targetAxis)
            paint(x, y);
        end
    end
    
    function stopPainting(~, ~)
        % Stop painting when the mouse button is released
        set(hFig, 'WindowButtonMotionFcn', '');
        set(hFig, 'WindowButtonUpFcn', '');
    end
    
    function paint(x, y)
        % Paint a circle of the specified brush size and color
        rectangle('Parent', targetAxis, 'Position', [x - brushSize/2, y - brushSize/2, brushSize, brushSize], ...
                  'Curvature', [1, 1], 'FaceColor', brushColor, 'EdgeColor', 'none');
    end

    function keyPressHandler(~, event)
        % Exit the interactive mode when 'q' is pressed
        if strcmp(event.Key, 'q')
            close(hFig);
        end
    end

    function inside = isInsideAxis(x, y, ax)
        % Check if the (x, y) coordinate is within the bounds of the axis
        xLimits = get(ax, 'XLim');
        yLimits = get(ax, 'YLim');
        inside = (x >= xLimits(1) && x <= xLimits(2) && y >= yLimits(1) && y <= yLimits(2));
    end
end
