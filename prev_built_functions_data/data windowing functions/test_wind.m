function test_wind()
    [img_train,gt_train,img_valid,gt_valid,~,~]=data_loader();
    gt_train(find(gt_train))=1;
    gt_valid(find(gt_valid))=1;
    % form smaller image batch
    [wind_img_train,wind_gt_train]=windowing_function(img_train,gt_train);

    load("trained_prototype4.mat")
    model=trained_prototype4;
    reconstruct_out=reconstruct_window(wind_img_train,model);

end