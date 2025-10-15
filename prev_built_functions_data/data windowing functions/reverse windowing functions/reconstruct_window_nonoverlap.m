function reconstruct_out=reconstruct_window_nonoverlap(orig_img_data,wind_img_data,model)
    % generate outcome variable
    outcome=activations(model,wind_img_data,'fb1m1a');

    % conduct mosiac fusion
    % setup outcome variable
    [row_size,col_size,num_img]=size(orig_img_data);
    reconstruct_out=zeros(row_size,col_size,num_img,class(outcome));
    step_size=row_size/8;
    % begin reconstruction
    for i_num=1:num_img
        pointer=1;                    
        for i_row=1:step_size:row_size
            for i_col=1:step_size:col_size
                % grab out current window
                current_wind=outcome(:,:,pointer,i_num);
                reconstruct_out(i_row:i_row+(step_size-1),i_col:i_col+(step_size-1),i_num)=current_wind;
                pointer=pointer+1;
            end
        end
    end
end