clear;
close all;
clc;
comInit;


%% init

p1 = struct;
p1.id = "test-example";

p1.uid = "jonghwan";
p1.eid = "220324-example";
p1.did = "SPECTRALIS_-_OCTA_C_1";
did = "SPECTRALIS_-_OCTA_C_1_Angio";

p1.pathraw = sprintf("%s/%s/%s", pathraw0, p1.uid, p1.eid);
p1.pathdata = sprintf("%s/%s/%s", pathdata0, p1.uid, p1.eid);
p1.pathrepo = sprintf("%s/test_load #%s", pathrepo0, p1.id);

if isempty(dir(p1.pathdata))
    mkdir(p1.pathdata);
end


%% load

r1 = struct;
fpath = sprintf("%s/%s.vol", p1.pathraw, p1.did);
vpath = sprintf("%s #vid.avi", p1.pathrepo);
[ r1.hdr, r1.hdrBscan, r1.slo, II, fig, figSeg, vid ] = ...
    ReadVOL_22URI( fpath, [], true, true, true, 1e4, p1.id, vpath );
SaveFig(figSeg, false, "#seg", sprintf("%s #seg", p1.pathrepo));

fpath = sprintf("%s/%s_Angio.vol", p1.pathraw, p1.did);
vpath = sprintf("%s #vidAng.avi", p1.pathrepo);
[ r1.hdrAng, r1.hdrBscanAng, r1.sloAng, DD, figAng, figSegAng, vidAng ] = ...
    ReadVOL_22URI( fpath, [], true, true, true, 1e3, sprintf("%s : ang", p1.id), vpath );
SaveFig(figSegAng, false, "#segAng", sprintf("%s #segAng", p1.pathrepo));

DD = min(DD,1);  % some pixels have 3.4e38


%% fig

figure;  imagesc(max(DD,[],3));  axis image;  grid on;

figure;  imshow(imrotate( squeeze(max(DD(75:150,:,:),[],1)), 90 )); 
% figure;  imshow(squeeze(max(DD(75:150,:,:),[],1))');  % same to above
