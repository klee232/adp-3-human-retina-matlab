

function [en_face_vessel_data_storage]=data_en_face_feature_generator(en_face_data_storage)

    %% conduct skeletionization 
    [skeleton_en_face_vessel_data_storage]=image_feature_en_face_generator_skeletion(en_face_data_storage);


    %% conduct nodes extraction
    [nodes_en_face_vessel_data_storage]=image_feature_en_face_generator_node(skeleton_en_face_vessel_data_storage);


    %% compute vessel branch length and tortuosity
    [en_face_vessel_length_data_storage, en_face_vessel_tortuosity_data_storage]=image_feature_en_face_generator_vessel_length_tortuosity(en_face_data_storage, nodes_en_face_vessel_data_storage);


    %% compute vessel diameter
    [en_face_vessel_diameter_data_storage]=image_feature_en_face_generator_vessel_diameter(skeleton_en_face_vessel_data_storage, en_face_data_storage);


    %% concatenate all vessel features
    num_feature=3;
    en_face_vessel_data_storage=zeros(size(en_face_vessel_diameter_data_storage,1),size(en_face_vessel_diameter_data_storage,2),size(en_face_vessel_diameter_data_storage,3),(num_feature*size(en_face_vessel_diameter_data_storage,4)));
    num_sample=size(en_face_vessel_diameter_data_storage,4);
    for i_sample=1:num_sample
        current_vessel_length_data=en_face_vessel_length_data_storage(:,:,:,i_sample);
        current_vessel_tortuosity_data=en_face_vessel_tortuosity_data_storage(:,:,:,i_sample);
        current_vessel_diameter_data=en_face_vessel_diameter_data_storage(:,:,:,i_sample);
        en_face_vessel_data_storage(:,:,:,(i_sample-1)*num_feature+1)=current_vessel_length_data;
        en_face_vessel_data_storage(:,:,:,(i_sample-1)*num_feature+2)=current_vessel_tortuosity_data;
        en_face_vessel_data_storage(:,:,:,(i_sample-1)*num_feature+3)=current_vessel_diameter_data;
    end
    

    %% save the vessel length data
    save("~/data/klee232/processed_data/octa gt arrays/octa_vessel_gt_data_en_face.mat", "en_face_vessel_data_storage",'-v7.3');

end




