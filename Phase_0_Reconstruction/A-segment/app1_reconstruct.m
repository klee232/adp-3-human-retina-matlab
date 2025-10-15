%%%% Reconstruct OCTA images of human retina
%   Output: 
%       p1: key parameters of app1
%       r1: a few results from app1, including retinal layer planes,
%       segmented by Heidelberg's inner algorithm.
%       II: OCT image, averaged by Heidelberg's inner algorithm
%       DD: OCTA image, averaged by Heidelberg's inner algorithm

function [p1, r1, II1, DD1] = app1_reconstruct(uid, eid, did, p1suff)

    %   comment out this when using app1 is called in batch1 
    %{
    close all;  clear;  clc;
    uid = "jonghwan";
    eid = "101-1017-y1";  did = "555";  % first data
    % eid = "101-1008-y1";  did = "506";  % bad data
    % eid = "101-1010-y1";  did = "526";  % bad data
    % eid = "101-1011-y1";  did = "545";  % bad data
    % eid = "101-1070-y1";  did = "653";  % bad data
    p1suff = "p1a";  % suffix of p1id (Process 1 ID)
    %}

    %%%% 10. Init    
    % uid, eid, did, p1suff => p1
    sec10
    ClearExcept(false, {'p1'})
    % p1 = sec10(uid, eid, did, p1suff, opt, pathraw0, pathdata0, pathrepo0, pathtemp0);  

    %%%% 11. Reconstruct OCT and OCTA images. [5 min]
    % p1 => p1, r1, II1, DD1
    %   II and DD have values between [0 1].
    %   p1.hdrBscan contains the vendor-defined planes of ILM, RPE, NFL. BUT,
    %   these planes seem to be in (y,x) than (x,y), and nz-ILM matched II.
    %   It also contains SEG for 19 planes, including the 3 main layers. 
    %   SEG(:,:,1) = ILM, SEG(:,:,2) = RPE, SEG(:,:,3) = NFL.
    %   r1.slo and r1.sloAng contains SLO images.
    sec11
    ClearExcept(false, {'p1', 'r1', 'II1', 'DD1'})
    % [p1, r1, II1, DD1] = sec11(p1);

    %%%% 12. Heidelberg-segmented layers
    %   This produces r1.seg/ilm/rpe/nfl, adjusted so that we don't need to use
    %   "nz-" and transpose.  It also produces r1.lyr, 3D array of layer index.
    %   This section temporarily applies Gaussian filtering (std=1) to II and
    %   DD for report figures. The output II2 and DD2 are not affected by this.
    sec12
    ClearExcept(false, {'p1', 'r1', 'II1', 'DD1'})  % we don't need II2, DD2
    % [p1, r1] = sec12(p1, r1, II1, DD1);

    %%%% 13. Layer analysis
    %   This conducts simple analysis of layers, like mean intensity and
    %   decorrelation values.
    sec13
    ClearExcept(false, {'p1', 'r1', 'II1', 'DD1'})  % we don't need II3, DD3
    % [p1, r1] = sec13(p1, r1, II1, DD1);

    %%%% 19. Save
    sec19
    % sec19(p1, r1, II1, DD1);

end