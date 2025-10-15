% Created by Kuan-Min Lee
% Createed date: Aug. 16th, 2024

% Brief User Introduction:
% This function is built to create User Interface for mannual selection (positive) for
% octa image

% Input parameter:
% i_slice: current slice index
% current_series: oct ground truth generated from the previous phase (2D image array)
% oct_gt_storage: oct ground truth generated from the previous phase (But
% in 3D) (3D image array)
% oct_storage: original oct image (3D image array)


% Output:
% current_series: processed octa ground truth (2D image array)
% sign: call the driver to continue or stop (string)


function [current_series,sign]=image_groundtruth_refiner_oct_rev(i_slice,current_series,current_series_oct,oct_gt_storage,oct_storage)
    %% setup indicator
    sign="continue";

    %% setup manual selection window for gap filling
    % get the MIP of B-scan
    OCT_gt_B_scan_x=squeeze(max(oct_gt_storage,[],2));
    OCT_gt_B_scan_y=squeeze(max(oct_gt_storage,[],3));
    OCT_B_scan_x=squeeze(max(oct_storage,[],2));
    OCT_B_scan_y=squeeze(max(oct_storage,[],3));

    % set up rectangle property
    % Define the initial position and size of the rectangle
    rectWidth = 50;
    rectHeight = 50;
    initialPosition = [1, 1, rectWidth, rectHeight]; % [x, y, width, height]
   
    % set up selection window z
    figure;
    hold on

    % octa part
    ax1=subplot(3,4,1);
    img_title=strcat("OCT groundtruth Slice: ",string(i_slice));
    img_note="Please press w, a, s, d to move, and q for exit";
    % create rectangle object
    h_ax1img=imshow(current_series,'Parent',ax1);
    hRect=drawrectangle('Position', initialPosition, 'LineWidth', 2, 'Color', 'r');
    title(img_title);
    xlabel(img_note);
    % B-scan on x plane
    ax2=subplot(3,4,5);
    img_title=strcat("OCT B-scan on x plane groundtruth Slice: ",string(i_slice));
    img_note="Please press w, a, s, d to move, and q for exit";
    imshow(OCT_gt_B_scan_x,'Parent',ax2);
    title(img_title);
    xlabel(img_note);
    % B-scan on y plane
    ax3=subplot(3,4,9);
    img_title=strcat("OCT B-scan on y plane groundtruth Slice: ",string(i_slice));
    img_note="Please press w, a, s, d to move, and q for exit";
    imshow(OCT_gt_B_scan_y,'Parent',ax3);
    title(img_title);
    xlabel(img_note);

    % oct part
    ax4=subplot(3,4,2);
    img_title="OCT raw image";
    % Create a rectangle object on the image
    imshow(current_series_oct,'Parent',ax4);
    hRect_oct = drawrectangle('Position', initialPosition, 'LineWidth', 2, 'Color', 'r');
    title(img_title);
    % B-scan on x plane
    ax5=subplot(3,4,6);
    img_title="OCT B-scan raw image on x plane";
    imshow(OCT_B_scan_x,'Parent',ax5);
    title(img_title);
    % B-scan on y plane
    ax6=subplot(3,4,10);
    img_title="OCT B-scan raw image on y plane";
    imshow(OCT_B_scan_y,'Parent',ax6);
    title(img_title);

    % cropped region part for OCTA
    ax7=subplot(3,4,[3,7,11]);
    pos=hRect.Position;
    croppedRegion1 = imcrop(current_series, pos);
    h_ax7img=imshow(croppedRegion1);
    img_title='Cropped Region from OCT gt';
    title(img_title);
    xlabel('Left click to bleach, right click to darken, q to quit')

    % cropped regions part for OCT
    subplot(3,4,[4,8,12]);
    pos_oct=hRect_oct.Position;
    croppedRegion2 = imcrop(current_series_oct, pos_oct);
    h_ax8img=imshow(croppedRegion2);
    img_title='Cropped Region from OCT';
    title(img_title);


    %% window selection part
    counter=1;
    while true
        % get the current rectangle ROI positions
        OCT_ROI_pos=hRect.Position;
        OCT_ROI_pos_y=OCT_ROI_pos(2);
        OCT_ROI_pos_x=OCT_ROI_pos(1);
        OCT_ROI_pos_h=OCT_ROI_pos(3);
        OCT_ROI_pos_w=OCT_ROI_pos(4);
        % get the current series image size
        img_h=size(current_series,1);
        img_w=size(current_series,2);
        if counter == 1
           % launch image binarization brush
           [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax7img,current_series,croppedRegion1,hRect,ax7);
           set(h_ax1img, 'CData', current_series);
           set(h_ax7img, 'CData', croppedRegion1);
           counter=counter+1;
        else
            if waitforbuttonpress
                cc = get(gcf,'CurrentCharacter');                
                % press w button to move upward by the height of the window
                % (subtraction in y)
                if strcmp(cc,'w')
                    update_ROI_pos_y=OCT_ROI_pos_y-OCT_ROI_pos_h;
                    % avoid out of boundary circumstance
                    if update_ROI_pos_y<=0
                        update_ROI_pos_y=OCT_ROI_pos_y;
                    end
                    % upade current rectangle ROI position
                    hRect.Position(2)=update_ROI_pos_y;
                    hRect_oct.Position(2)=update_ROI_pos_y;
                    % update the display ROI image in ax3 and ax4
                    croppedRegion1=imcrop(current_series,hRect.Position);
                    croppedRegion2=imcrop(current_series_oct,hRect_oct.Position);
                    % Update the image data
                    set(h_ax7img, 'CData', croppedRegion1); 
                    set(h_ax8img, 'CData', croppedRegion2);
                    % launch image binarization brush
                    [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax7img,current_series,croppedRegion1,hRect,ax7);
                    % update current brushed outcome
                    set(h_ax1img, 'CData', current_series);
                    set(h_ax7img, 'CData', croppedRegion1);
    
                % press a button to move left by the width of the window
                % (subtraction in x)
                elseif strcmp(cc,'a')
                    update_ROI_pos_x=OCT_ROI_pos_x-OCT_ROI_pos_w;
                    % avoid out of boundary circumstance
                    if update_ROI_pos_x<=0
                        update_ROI_pos_x=OCT_ROI_pos_x;
                    end
                    % upade current rectangle ROI position
                    hRect.Position(1)=update_ROI_pos_x;
                    hRect_oct.Position(1)=update_ROI_pos_x;
                    % update the display ROI image in ax3 and ax4
                    croppedRegion1=imcrop(current_series,hRect.Position);
                    croppedRegion2=imcrop(current_series_oct,hRect_oct.Position);
                    % Update the image data
                    set(h_ax7img, 'CData', croppedRegion1); 
                    set(h_ax8img, 'CData', croppedRegion2);
                    % launch image binarization brush
                    [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax7img,current_series,croppedRegion1,hRect,ax7);
                    % update current brushed outcome
                    set(h_ax1img, 'CData', current_series);
                    set(h_ax7img, 'CData', croppedRegion1);
    
                % press s button to move down by the height of the window
                % (addition in y)
                elseif strcmp(cc,'s')
                    % update the current y position
                    update_ROI_pos_y=OCT_ROI_pos_y+OCT_ROI_pos_h;
                    % avoid out of boundary circumstance
                    if update_ROI_pos_y>img_h
                        update_ROI_pos_y=OCT_ROI_pos_y;
                    end
                    % upade current rectangle ROI position
                    hRect.Position(2)=update_ROI_pos_y;
                    hRect_oct.Position(2)=update_ROI_pos_y;
                    % update the display ROI image in ax3 and ax4
                    croppedRegion1=imcrop(current_series,hRect.Position);
                    croppedRegion2=imcrop(current_series_oct,hRect_oct.Position);
                    % Update the image data
                    set(h_ax7img, 'CData', croppedRegion1); 
                    set(h_ax8img, 'CData', croppedRegion2);
                    % launch image binarization brush
                    [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax7img,current_series,croppedRegion1,hRect,ax7);
                    % update current brushed outcome
                    set(h_ax1img, 'CData', current_series);
                    set(h_ax7img, 'CData', croppedRegion1);
    
                % press d button to move right
                elseif strcmp(cc,'d')
                    update_ROI_pos_x=OCT_ROI_pos_x+OCT_ROI_pos_w;
                    % avoid out of boundary circumstance
                    if update_ROI_pos_x>img_w
                        update_ROI_pos_x=OCT_ROI_pos_x;
                    end
                    % upade current rectangle ROI position
                    hRect.Position(1)=update_ROI_pos_x;
                    hRect_oct.Position(1)=update_ROI_pos_x;
                    % update the display ROI image in ax3 and ax4
                    croppedRegion1=imcrop(current_series,hRect.Position);
                    croppedRegion2=imcrop(current_series_oct,hRect_oct.Position);
                    % Update the image data
                    set(h_ax7img, 'CData', croppedRegion1); 
                    set(h_ax8img, 'CData', croppedRegion2);
                    % launch image binarization brush
                    [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax7img,current_series,croppedRegion1,hRect,ax7);
                    % update current brushed outcome
                    set(h_ax1img, 'CData', current_series);
                    set(h_ax7img, 'CData', croppedRegion1);
    
                % press t button to teminate current and store temp work
                elseif strcmp(cc,'t')
                    % create function indicator
                    sign="break";
                    indicator="octa-neg";
                    % save intermediate file
                    save("temp_data/octa_data_gt_ind.mat","i_slice");
                    save("temp_data/indicator.mat",'indicator');
                    break;
    
                % press q button to finish current slice segmentation
                elseif strcmp(cc,'q')
                    break;
                end
            end % end of press button 
        end % end of counter
    end % end of while loop
end
                

%% subfunction for brushing cropped regions
function [current_series,croppedRegion1]=image_groundtruth_refiner_oct_brush(h_ax1img,h_ax3img,current_series,croppedRegion1,h_Rect,hAx_crop1)
    while true
        %% retrieve input and create image brush
        [~, ~, button] = ginput(1);
        % make sure the mouse position is within the target axis
        targetAxis = hAx_crop1;
        hold(targetAxis, 'on');
        % Get the position of the mouse click
        mousePos=get(targetAxis, 'CurrentPoint');
        mouse_x=round(mousePos(1, 1));
        mouse_y=round(mousePos(1, 2));
        % set the brush size
        brush_size=3;
        % paint the image only within region of interest
        % Check if the (x, y) coordinate is within the bounds of the axis
        xLimits=get(targetAxis, 'XLim');
        yLimits=get(targetAxis, 'YLim');
        inside_ind=(mouse_x>=xLimits(1) && mouse_x<=xLimits(2) && mouse_y>=yLimits(1) && mouse_y<=yLimits(2));
        % create a mask for selected region
        start_x=max(1,mouse_x-brush_size);
        start_y=max(1,mouse_y-brush_size);
        end_x=min(size(croppedRegion1,2),mouse_x+brush_size);
        end_y=min(size(croppedRegion1,1),mouse_y+brush_size);
        [img_brush_x, img_brush_y]=meshgrid(start_x:end_x, start_y:end_y);
        center=[mouse_x,mouse_y];
        img_brush_dist=sqrt((img_brush_x-center(1)).^2 + (img_brush_y-center(2)).^2);
        img_brush = img_brush_dist<brush_size;

        %% click to start brushing
        if button==1
            if inside_ind
                % make the region within the brush 1s
                croppedRegion1(start_y:mouse_y+brush_size,start_x:mouse_x+brush_size)=...
                    croppedRegion1(start_y:mouse_y+brush_size,start_x:mouse_x+brush_size)+img_brush;
                croppedRegion1(croppedRegion1>=1)=1;
                % update current series
                h_Rect_x=round(h_Rect.Position(1));
                h_Rect_y=round(h_Rect.Position(2));
                h_Rect_w=round(h_Rect.Position(3));
                h_Rect_h=round(h_Rect.Position(4));
                current_series(h_Rect_y:h_Rect_y+h_Rect_h,h_Rect_x:h_Rect_x+h_Rect_w)=croppedRegion1;
                set(h_ax1img, 'CData', current_series);
                set(h_ax3img, 'CData', croppedRegion1);
            end
        % right click to make it dark
        elseif button==3
                % make the region within the brush 1s
                croppedRegion1(start_y:mouse_y+brush_size,start_x:mouse_x+brush_size)=...
                    croppedRegion1(start_y:mouse_y+brush_size,start_x:mouse_x+brush_size).*img_brush;
                croppedRegion1(croppedRegion1>=1)=1;
                % update current series
                h_Rect_x=round(h_Rect.Position(1));
                h_Rect_y=round(h_Rect.Position(2));
                h_Rect_w=round(h_Rect.Position(3));
                h_Rect_h=round(h_Rect.Position(4));
                current_series(h_Rect_y:h_Rect_y+h_Rect_h,h_Rect_x:h_Rect_x+h_Rect_w)=croppedRegion1;
                set(h_ax1img, 'CData', current_series);
                set(h_ax3img, 'CData', croppedRegion1);

        % press q to quit the current function
        elseif button==113
            return;
        end
    end
end


