

function [fuse_data_storage]=image_groundtruth_generator_en_face_fuse(denoised_data_storage, edge_data_storage, frangi_data_storage)

    %% fuse the feature together
    fuse_data_storage=(denoised_data_storage+edge_data_storage).*(1-frangi_data_storage);

end