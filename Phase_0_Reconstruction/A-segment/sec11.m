% function [p1, r1, II1, DD1] = sec11(p1)
disp("SECTION 11 RUNNING ...");
r1 = struct;

%%%% Load and reconstruct

% fpath = sprintf("%s/ARIAS_B_%s.vol", p1.pathraw, p1.did);
fpath = sprintf("%s/*_%s.vol", p1.pathraw, p1.did);
d = dir(fpath);
if numel(d) ~= 1
    error("Raw file error:\n%s", fpath);
end
fid = sprintf("%s | %s", p1.eid, p1.id);  % for fig and figSeg
fpath = sprintf("%s/%s", d.folder, d.name);
vpath = sprintf("%s #11-vid.avi", p1.pathrepo);  % save video of 3D
[ p1.hdr, p1.hdrBscan, r1.slo, II1, fig, figSeg, vid ] = ...
    ReadVOL_22URI( fpath, [], true, true, true, p1.opt.snr, fid, vpath );
SaveFig(fig, false, "#11-recon", sprintf("%s #11-recon", p1.pathrepo));
SaveFig(figSeg, false, "#11-seg", sprintf("%s #11-seg", p1.pathrepo));

% fpath = sprintf("%s/ARIAS_B_%s_Angio.vol", p1.pathraw, p1.did);
fpath = sprintf("%s/*_%s_Angio.vol", p1.pathraw, p1.did);
d = dir(fpath);
if numel(d) ~= 1
    error("Raw file error:\n%s", fpath);
end
fpath = sprintf("%s/%s", d.folder, d.name);
vpath = sprintf("%s #11-vidAng.avi", p1.pathrepo);
[ p1.hdrAng, p1.hdrBscanAng, r1.sloAng, DD1, figAng, figSegAng, vidAng ] = ...
    ReadVOL_22URI( fpath, [], true, true, true, p1.opt.snrAng, sprintf("%s : ang", fid), vpath );
SaveFig(figAng, false, "#11-reconAng", sprintf("%s #11-reconAng", p1.pathrepo));
SaveFig(figSegAng, false, "#11-segAng", sprintf("%s #11-segAng", p1.pathrepo));

DD1 = min(DD1,1);  % some pixels have 3.4e38

%%%% fig

% fid = 1-reconstruct #eid #p1id #11-preview
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #11-preview", fid);
fig = NewFig2(3,3);  colormap(gray);
% img = squeeze(max(DD1, [], 1));  % max make most voxels to 1
img = squeeze(mean(DD1, 1));
if p1.opt.snrAng > 0
    img = Rescale(log10(img), [0 1], log10([1/p1.opt.snrAng 1]));
else
    % img with max has many 1's, so bottom 1% is 1.
    % img = min(max(img, prctile(img, 10, 'all')), prctile(img, 99, 'all'));
    img = imadjust(img);
end
imagesc(img');
ax = gca;
ax.YDir = "normal";
ax.XLabel.String = "X";
ax.YLabel.String = "Y";
ax.Title.String = fid;
SaveFig(fig, false, fid, sprintf("%s #11-preview", p1.pathrepo));

disp("SECTION 11 COMPLETED.");