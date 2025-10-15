function max_mask = find_all_max_indices(input_feat,filt_size)
    % retrieve the shape of input image
    [row_in,col_in]=size(input_feat);

    % Reshape the input feature 4*num of regions
    reshaped_input = im2col(input_feat, [filt_size filt_size],"distinct");

    % create max mask matrix
    reshaped_max_mask=zeros(size(reshaped_input));

    % retrieve the maximum value 
    [maximums,~]=max(reshaped_input,[],1,"linear");
    
    % retireve the maximum linear indices
    all_max_inds=reshaped_input==maximums;

    % assign those values as 1 to max mask matrix
    reshaped_max_mask(all_max_inds)=1;

    % count the number of ones in each column
    num_ones=sum(reshaped_max_mask==1,1);

    % average it along first dimension
    reshaped_max_mask=reshaped_max_mask./num_ones;

    % reshape back
    max_mask=col2im(reshaped_max_mask,[filt_size filt_size],[row_in col_in],'distinct');

end