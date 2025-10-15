function reconstruct_out=reconstruct_window(orig_img_data,wind_img_data,model)
    % generate outcome variable
    outcome=activations(model,wind_img_data,'fb1m1a');

    % conduct mosiac fusion
    % setup outcome variable
    [row_size,col_size,num_img]=size(orig_img_data);
    reconstruct_out=zeros(row_size,col_size,num_img,class(outcome));
    step_size=row_size/16;
    % begin reconstruction
    for i_num=1:num_img
        pointer=1;                    
        for i_row=1:step_size:((row_size-row_size/8)+1)
            for i_col=1:step_size:((col_size-col_size/8)+1)
                % grab out current window
                current_wind=outcome(:,:,pointer,i_num);
                % if this is the first image, store it directly
                if (i_row==1) && (i_col==1)
                    reconstruct_out(i_row:(i_row+37),i_col:(i_col+37),i_num)=current_wind;
                % if not, conduct mosiac image fusion
                else
                    % grab out region for both image fusion and none-image
                    % fustion
                    % region in row
                    row_index=i_row:i_row+(size(current_wind,1)-1);
                    col_index=i_col:i_col+(size(current_wind,2)-1);
                    row_index_wind=1:size(current_wind,1);
                    col_index_wind=1:size(current_wind,2);
                    if i_row~=1
                        row_nonfusion=i_row+step_size:i_row+size(current_wind,1)-1;
                        row_fusion=setdiff(row_index,row_nonfusion);
                        row_nonfusion_wind=(size(current_wind,1)/2+1):size(current_wind,1);
                        row_fusion_wind=setdiff(row_index_wind,row_nonfusion_wind);
                    else
                        row_nonfusion=row_index;
                        row_fusion=row_index;
                        row_nonfusion_wind=row_index_wind;
                        row_fusion_wind=row_index_wind;
                    end
                    if i_col~=1
                        col_nonfusion=i_col+step_size:i_col+size(current_wind,2)-1;
                        col_fusion=setdiff(col_index,col_nonfusion);
                        col_nonfusion_wind=(size(current_wind,2)/2+1):size(current_wind,2);
                        col_fusion_wind=setdiff(col_index_wind,col_nonfusion_wind);                              
                    else
                        col_nonfusion=col_index;
                        col_fusion=col_index;
                        col_nonfusion_wind=col_index_wind;
                        col_fusion_wind=col_index_wind;
                    end
                    % conduct fusion in nonfusion region (directly storing)
                    reconstruct_out(row_nonfusion,col_nonfusion)=current_wind(row_nonfusion_wind,col_nonfusion_wind);
                    % conduct fusion in fusion region (maximum pooling)
                    fusion_region=reconstruct_out(row_fusion,col_fusion);
                    fusion_window=current_wind(row_fusion_wind,col_fusion_wind);
                    fusion_out=max(fusion_region,fusion_window);
                    reconstruct_out(row_fusion,col_fusion)=fusion_out;
                end
                pointer=pointer+1;
            end
        end
    end
end
