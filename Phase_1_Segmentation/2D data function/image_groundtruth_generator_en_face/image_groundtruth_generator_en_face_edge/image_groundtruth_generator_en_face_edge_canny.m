

function [edge_data_storage]=image_groundtruth_generator_en_face_edge_canny(data_storage)

    %% conduct canny edge image 
    num_file=size(data_storage,1);


    edge_data_storage=zeros(size(data_storage));
    for i_file=1:num_file
        current_data_img=squeeze(data_storage(i_file,:,:));
        edge_current_data_img=edge(current_data_img,'canny',0.1);
        edge_data_storage(i_file,:,:)=edge_current_data_img;
    end

end