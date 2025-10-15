% Created by Kuan-Min Lee
% Createed date: May 23rd, 2024

% Brief User Introduction:
% This function is built to create the binary groundtruth for the training
% 3D OCTA image network. The below function contains two parts: image
% denoising part and segmentation paprt

% Input parameter:
% OCTA_img: input OCTA training image
% OCT_img: input OCT training image

% Output:
% OCTA_img_gt: groundturth for OCTA image
% OCT_img_gt: grountruth for OCT image
% picture_obj: struct object for storing information

function [OCTA_img_gt, OCT_img_gt,OCTA_img_den,OCT_img_den, picture_obj]=image_groundtruth_generator(OCTA_img,OCT_img)
    % create struct object for storing information
    picture_obj=struct;
    picture_obj.OCTA_img=OCTA_img;  % store OCTA image
    picture_obj.OCT=OCT_img;

    % create struct object for image denoising processing type
    opt=struct;
    opt.oSeg = "gthr";
    opt.filt = "median";
    opt.filt_sz = [11 11 11];
    % opt.thres_ratio = 0.8;

    % conduct image denoising
    [filtered_OCTA_img]=image_region_filter(OCTA_img);
    [picture_obj,OCTA_img_den]=image_denoise(picture_obj,filtered_OCTA_img,opt); % OCTA image
    [filtered_OCT_img]=image_region_filter(OCT_img);
    [picture_obj,OCT_img_den]=image_denoise(picture_obj,filtered_OCT_img,opt); % OCT image
    % store information in picture object
    picture_obj.OCTA_img_den=OCTA_img_den;
    picture_obj.OCT_img_den=OCT_img_den;

    % conduct image segmentation
    [OCTA_img_gt]=image_3D_edge_binarizer(OCTA_img_den);
    [OCT_img_gt]=image_3D_edge_binarizer(OCT_img_den);
    % store informationn in picture object
    picture_obj.OCTA_img_gt=OCTA_img_gt;
    picture_obj.OCT_img_gt=OCT_img_gt;

end