function [new_imageData]=interactiveBrushTool(imageData)
    % Display the image
    hFig = figure('Name', 'Interactive Brush Tool', ...
                  'WindowButtonDownFcn', @startBrushing, ...
                  'WindowButtonUpFcn', @stopBrushing, ...
                  'WindowButtonMotionFcn', @brush);

    hAx = axes('Parent', hFig);
    hImg = imshow(imageData, 'Parent', hAx);
    hold on;

    

    
new_imageData=imageData;
end