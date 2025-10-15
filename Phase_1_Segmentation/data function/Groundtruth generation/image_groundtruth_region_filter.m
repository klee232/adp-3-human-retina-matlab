% Created by Kuan-Min Lee
% Createed date: Jul. 30th, 2024

% Brief User Introduction:
% The following function is utilized to filter out the unwanted region of
% the OCTA image

% Input parameter:
% DD: original OCTA image

% Output:
% DD_filtered: filtered OCTA image


function [picture_obj_octa,picture_obj_oct,DD_filtered,mask_filtered]=image_groundtruth_region_filter(DD,II,mask_seg)
    
    %% create storage struct
    picture_obj_octa=struct;
    picture_obj_oct=struct;


    %% retireve the image dimension (DD here for OCTA and DD and II (OCT) is supposed to have the same dimension)
    num_slice=size(DD,1);
    num_row=size(DD,2);
    num_col=size(DD,3);


    %% plot the gradient plot along, z, x, and y directions
    % create mean image arrays for each z location (OCTA)
    pixel_mean_z=mean(DD,[2,3]); % mean intensity along z direction
    pixel_mean_z=squeeze(pixel_mean_z);
    % create mean image arrays for each z location (OCT)
    pixel_mean_z_oct=mean(II,[2,3]); % mean intensity along z direction
    pixel_mean_z_oct=squeeze(pixel_mean_z_oct);


    %% create MIP image for display
    % OCTA
    mip_DD_z_x=squeeze(max(DD,[],3));
    mip_DD_z_y=squeeze(max(DD,[],2));
    % OCT
    mip_II_z_x=squeeze(max(II,[],3));
    mip_II_z_y=squeeze(max(II,[],2));


    %% create manual selection window for filtering out region
    z_ind=[0 0];
    if ~any(z_ind)
        z_ind = [1 num_slice];  % the initial: remove the autocorrelation noise at the top of B-scan
    end
    while true
        % selection window z
        % OCTA
        % MIP on x axis
        figure(1);
        t = tiledlayout(3, 2); % Create a 1x2 tiled layout
        % Adjust the spacing and padding
        t.TileSpacing = 'compact'; % Options: 'compact', 'none', 'loose'
        t.Padding = 'compact';     % Options: 'compact', 'none', 'loose'
        ax1=nexttile;
        imagesc(mip_DD_z_x);
        colormap("gray");
        line([1 num_row], [1 1]*z_ind(1), Color='r');  
        line([1 num_row], [1 1]*z_ind(2), Color='r');
        xlabel("x");
        ylabel("z");
        title("MIP for OCTA on X plane");
        pos1 = ax1.Position;
        pos1(3) = pos1(3) * 1.2; 
        pos1(4) = pos1(4) * 1.2; 
        set(ax1, 'Position', pos1);
        % MIP on y axis
        ax2=nexttile;
        imagesc(mip_DD_z_y);
        colormap("gray");
        line([1 num_col], [1 1]*z_ind(1), Color='r');  
        line([1 num_col], [1 1]*z_ind(2), Color='r');
        title('MIP for OCTA on Y plane')
        pos2 = ax2.Position;
        pos2(3) = pos2(3) * 1.2; % Decrease width by 20%
        pos2(4) = pos2(4) * 1.2; % Decrease width by 20%
        set(ax2, 'Position', pos2);
        % OCT
        % MIP on x axis
        ax3=nexttile;
        imagesc(mip_II_z_x);
        colormap("gray");
        line([1 num_row], [1 1]*z_ind(1), Color='r');  
        line([1 num_row], [1 1]*z_ind(2), Color='r');
        title('MIP for OCT on X plane')
        pos3 = ax3.Position;
        pos3(3) = pos3(3) * 1.2; % Decrease width by 20%
        pos3(4) = pos3(4) * 1.2; % Decrease width by 20%
        set(ax3, 'Position', pos3);
        % MIP on y axis
        ax4=nexttile;
        imagesc(mip_II_z_y);
        colormap("gray");
        line([1 num_col], [1 1]*z_ind(1), Color='r');  
        line([1 num_col], [1 1]*z_ind(2), Color='r');
        title('MIP for OCT on Y plane')
        pos4 = ax4.Position;
        pos4(3) = pos4(3) * 1.2; % Decrease width by 20%
        pos4(4) = pos4(4) * 1.2; % Decrease width by 20%
        set(ax4, 'Position', pos4);

        % Window Selection part
        ax5=nexttile([1, 2]); % Span 2 rows and 1 column
        % subplot(3,2,[5,6]);
        octa_z=plot(1:num_slice, pixel_mean_z, Linewidth=2,Color='b');
        hold on
        oct_z=plot(1:num_slice, pixel_mean_z_oct, Linewidth=2,Color='g');
        axis tight;
        grid on;
        ax = gca;
        line([1 1]*z_ind(1), ax.YLim, Color='r');
        line([1 1]*z_ind(2), ax.YLim, Color='r');
        line([1 num_slice], [1 1]*pixel_mean_z(z_ind(1)));
        line([1 num_slice], [1 1]*pixel_mean_z(z_ind(2)));
        legend([octa_z,oct_z],'OCTA','OCT');
        xlabel('z')
        ylabel('Pixel mean in depth direction')
        title('Select z for cropping here')
        hold off
        pos5 = ax5.Position;
        pos5(3) = pos5(3) * 0.8; % Decrease width by 20%
        set(ax5, 'Position', pos5);
        if waitforbuttonpress
            cc = get(gcf,'CurrentCharacter');
            if strcmp(cc,'q')
                break;
            elseif strcmp(cc,'g')
                [z,~] = ginput(2);
                z_ind = round(z)';
            end
        end 
    end
    saveas(gcf,'z-direction_plot','png')


    %% set up x and y cropping indices
    thres_boundary=10;
    start_ind=thres_boundary;
    end_ind_x=num_row-thres_boundary;
    end_ind_y=num_col-thres_boundary;
    x_ind=[start_ind end_ind_x];
    y_ind=[start_ind end_ind_y];
    

    %% filter out the region that is not within the object
    % OCTA
    DD_filtered_xy=DD(:,x_ind(1):x_ind(2),y_ind(1):y_ind(2));
    DD_filtered=DD(z_ind(1):z_ind(2),x_ind(1):x_ind(2),y_ind(1):y_ind(2));
    % OCT
    II_filtered_xy=II(:,x_ind(1):x_ind(2),y_ind(1):y_ind(2));
    II_filtered=II(z_ind(1):z_ind(2),x_ind(1):x_ind(2),y_ind(1):y_ind(2));
    % mask-seg
    mask_filtered_xy=mask_seg(:,x_ind(1):x_ind(2),y_ind(1):y_ind(2));
    mask_filtered=mask_seg(z_ind(1):z_ind(2),x_ind(1):x_ind(2),y_ind(1):y_ind(2));


    %% display final MIP image
    % OCTA
    MIP_DD_filtered_z=squeeze(max(DD_filtered,[],1));
    MIP_DD_filtered_x=squeeze(max(DD_filtered,[],3));
    MIP_DD_filtered_y=squeeze(max(DD_filtered,[],2));
    figure(2);
    subplot(131);
    imagesc(MIP_DD_filtered_z);
    colormap("gray");
    title("Maximum Intensity Plot onto z plane")
    subplot(132);
    imagesc(MIP_DD_filtered_x);
    colormap("gray");
    title("Maximum Intensity Plot onto x plane")
    subplot(133);
    imagesc(MIP_DD_filtered_y);
    colormap("gray");
    title("Maximum Intensity Plot onto y plane")
    saveas(gcf,'octa_bscan_plot','png')

    % OCT
    MIP_II_filtered_z=squeeze(max(II_filtered,[],1));
    MIP_II_filtered_x=squeeze(max(II_filtered,[],3));
    MIP_II_filtered_y=squeeze(max(II_filtered,[],2));
    figure(3);
    subplot(131);
    imagesc(MIP_II_filtered_z);
    colormap("gray");
    title("Maximum Intensity Plot onto z plane")
    subplot(132);
    imagesc(MIP_II_filtered_x);
    colormap("gray");
    title("Maximum Intensity Plot onto x plane")
    subplot(133);
    imagesc(MIP_II_filtered_y);
    colormap("gray");
    title("Maximum Intensity Plot onto y plane")
    saveas(gcf,'oct_bscan_plot','png')    

    %% store picture object
    % cropping index
    picture_obj_octa.z_ind=z_ind;
    picture_obj_octa.y_ind=y_ind;
    picture_obj_octa.x_ind=x_ind;
    picture_obj_oct.z_ind=z_ind;
    picture_obj_oct.y_ind=y_ind;
    picture_obj_oct.x_ind=x_ind;

    % filtered image
    picture_obj_octa.filtered_img=DD_filtered;
    picture_obj_octa.filtered_img_xy=DD_filtered_xy;
    picture_obj_oct.filtered_img=II_filtered;
    picture_obj_oct.filtered_img_xy=II_filtered_xy;

    % filtered mask
    picture_obj_octa.filtered_mask_seg=mask_filtered;
    picture_obj_octa.filtered_mask_seg_xy=mask_filtered_xy;
    picture_obj_oct.filtered_mask_seg=mask_filtered;
    picture_obj_oct.filtered_mask_seg_xy=mask_filtered_xy;

    
    %% save the filtered image and struct
    save("temp_data/picture_obj_octa.mat","picture_obj_octa");
    save("temp_data/picture_obj_oct.mat","picture_obj_oct");

end