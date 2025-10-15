function [p1, r1, II, DD] = sec12(p1, r1, II, DD)

disp("SECTION 12 RUNNING ...")

[nz, nx, ny] = size(II);
II = imgaussfilt3(II, [1 1 1]);  % just for figure
DD = imgaussfilt3(DD, [1 1 1]);  % just for figure
DD0 = DD;  % for en face comparison


%% Check p1.hdrBscan.SEG

seg = p1.hdrBscan.SEG;
ilm = p1.hdrBscan.ILM;
rpe = p1.hdrBscan.RPE;
nfl = p1.hdrBscan.NFL;
segAng = p1.hdrBscanAng.SEG;
ilmAng = p1.hdrBscanAng.ILM;
rpeAng = p1.hdrBscanAng.RPE;
nflAng = p1.hdrBscanAng.NFL;

% difference between OCT and OCTA data
diffIlm = mean(abs(ilm - ilmAng), 'all');
diffRpe = mean(abs(rpe - rpeAng), 'all');
diffNfl = mean(abs(nfl - nflAng), 'all');
msg = sprintf("ILM difference between p1.hdrBscan and p1.hdrBscanAng: %.2f vox", diffIlm);
if diffIlm > 1
    warning(msg);
else
    disp(msg);
end
msg = sprintf("RPE difference between p1.hdrBscan and p1.hdrBscanAng: %.2f vox", diffRpe);
if diffRpe > 1
    warning(msg);
else
    disp(msg);
end
msg = sprintf("NFL difference between p1.hdrBscan and p1.hdrBscanAng: %.2f vox", diffNfl);
if diffNfl > 1
    warning(msg);
else
    disp(msg);
end

% check SEG's 1-3 planes
diffSeg = mean(abs(seg(:,:,1) - ilm), 'all');
if diffSeg < 0.01
    fprintf("SEG(:,:,1) = ILM: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,1) ~= ILM: diff = %.2f vox", diffSeg);
end
diffSeg = mean(abs(seg(:,:,2) - rpe), 'all');
if diffSeg < 0.01
    fprintf("SEG(:,:,2) = RPE: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,2) ~= RPE: diff = %.2f vox", diffSeg);
end
diffSeg = mean(abs(seg(:,:,3) - nfl), 'all');
if diffSeg < 0.01
    fprintf("SEG(:,:,3) = NFL: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,3) ~= NFL: diff = %.2f vox", diffSeg);
end


%% plot p1.hdrBscan.SEG

fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-seg", fid);

y = round(ny * [.25 .5 .75]);

clr = lines;
fig = NewFig(4,5);  colormap(gray);
for iy=1:3
    y1 = y(iy);
    subplot(1,3,iy);
    img = II(:,:,y1);
    % imagesc(imadjust(img));
    imagesc(log(max(img,1e-4)));
    % axis image;
    for is=1:size(seg,3)
        % seg1 = seg(:,:,is);
        seg1 = nz - seg(:,:,is);
        if is <= 3  % ILM, RPE, NFL
            lw = 1.5;
        else 
            lw = 0.5;  % default of line()
        end
        line(1:nx, seg1(y1,:), Color=clr(is,:), LineWidth=lw);
    end
    ax = gca;
    ax.Title.String = sprintf("y=%d/%d", y1, ny);
end
sgtitle(fid);
SaveFig(fig, false, fid, sprintf("%s #12-seg", p1.pathrepo));


%% Set zero with a 3D mask

msk = ones(nz, nx, ny);  % mask

% set all voxels above ILM to zero
if p1.opt.bZeroAboveIlm
    ilm1 = repmat(reshape(ilm', [1 nx ny]), [nz 1 1]);
    [gz, gx, gy] = ndgrid(1:nz, 1:nx, 1:ny);
    msk(gz < (nz-ilm1)) = 0;
end

% set all voxels below RPE to zero
if p1.opt.bZeroBelowRpe
    rpe1 = repmat(reshape(rpe', [1 nx ny]), [nz 1 1]);
    if ~exist("gz", 'var')
        [gz, gx, gy] = ndgrid(1:nz, 1:nx, 1:ny);
    end
    msk(gz > (nz-rpe1)) = 0;
end

% plot mask
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-mask", fid);
fig = NewFig(4,5);  colormap(gray);
for iy=1:3
    y1 = y(iy);
    subplot(1,3,iy);
    imagesc(msk(:,:,y1));
    % axis image;
    for is=1:3  % ILM, RPE, NFL
        % seg1 = seg(:,:,is);
        seg1 = nz - seg(:,:,is);
        lw = 1.5;
        line(1:nx, seg1(y1,:), Color=clr(is,:), LineWidth=lw);
    end
    ax = gca;
    ax.Title.String = sprintf("y=%d/%d", y1, ny);
    if iy == 1
        ax.XLabel.String = [sprintf("bZeroAboveIlm=%d", p1.opt.bZeroAboveIlm), ...
            sprintf("bZeroBelowRpe=%d", p1.opt.bZeroBelowRpe)];
    end        
end
sgtitle(fid);
SaveFig(fig, false, fid, sprintf("%s #12-mask", p1.pathrepo));

II = II .* msk;
DD = DD .* msk;


%% Crop

zz = [1 nz];
zz(1) = min(nz - ilm, [], 'all') - p1.opt.zCropMargin;
zz(2) = max(nz - rpe, [], 'all') + p1.opt.zCropMargin;
if p1.opt.bKeepBelowRpe
    zz(2) = zz(2) + floor(mean(ilm-rpe, 'all'));
end
zz = min(max(round(zz), 1), nz);

% r1.seg has been applied "nz-".
seg = nz - seg;
ilm = nz - ilm;
rpe = nz - rpe;
nfl = nz - nfl;
r1.seg = seg - (zz(1)-1);
r1.ilm = ilm' - (zz(1)-1);  % transpose
r1.rpe = rpe' - (zz(1)-1);
r1.nfl = nfl' - (zz(1)-1);
for is=1:size(r1.seg,3)
    r1.seg(:,:,is) = r1.seg(:,:,is)';
end

% plot crop
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-crop", fid);
fig = NewFig(4,5);  colormap(gray);

subplot(131);
% img = log10(max(imadjust(max(II, [], 3)), 1e-4));
% img = log10(max(imadjust(sum(II, 3)), 1e-4));
% img = imadjust(sum(II, 3));
img = log(sum(II,3));
imagesc(img);
line([1 nx], [1 1]*zz(1), Color='r');
line([1 nx], [1 1]*zz(2), Color='r');
ax = gca;
ax.Title.String = "log sum of OCT";
ax.XLabel.String = sprintf("zz = %s", mat2str(zz));

subplot(132);
% img = log10(max(imadjust(max(DD, [], 3)), 1e-4));
% img = log10(max(imadjust(sum(DD, 3)), 1e-4));
% img = imadjust(sum(DD, 3));
img = log(sum(DD,3));
imagesc(img);
line([1 nx], [1 1]*zz(1), Color='r');
line([1 nx], [1 1]*zz(2), Color='r');
ax = gca;
ax.Title.String = "log sum of OCTA";

% crop
II = II(zz(1):zz(2),:,:);
DD = DD(zz(1):zz(2),:,:);

% iy=ny/2 after crop
subplot(133);
img = imadjust(II(:,:,end/2));
imagesc(img);
for is=1:3
    line(1:nx, r1.seg(:,end/2,is), Color=clr(is,:), LineWidth=1.5);
end
ax = gca;
ax.Title.String = sprintf("OCT at iy=%d/%d", ny/2, ny);

sgtitle(fid);
SaveFig(fig, false, fid, sprintf("%s #12-crop", p1.pathrepo));


%% en face DD

fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-compare", fid);
fig = NewFig(3,5);  colormap(gray);

subplot(121);
img = imadjust(squeeze(max(DD0, [], 1)));
% img = imadjust(squeeze(sum(DD0, 1)));
imagesc(img');
ax = gca;
ax.YDir = "normal";
ax.DataAspectRatio = [1 1 1];
ax.XTick = [];  ax.YTick = [];
ax.Title.String = "Original";

subplot(122);
img = imadjust(squeeze(max(DD, [], 1)));
% img = imadjust(squeeze(sum(DD, 1)));
imagesc(img');
ax = gca;
ax.YDir = "normal";
ax.DataAspectRatio = [1 1 1];
ax.XTick = [];  ax.YTick = [];
ax.Title.String = "After zeroing and cropping";
ax.XLabel.String = [sprintf("bZeroAboveIlm=%d", p1.opt.bZeroAboveIlm), ...
    sprintf("bZeroBelowRpe=%d", p1.opt.bZeroBelowRpe), ...
    sprintf("bKeepBelowRpe=%d", p1.opt.bKeepBelowRpe), ...
    ];

sgtitle(fid);
SaveFig(fig, false, fid, sprintf("%s #12-compare", p1.pathrepo));


disp("SECTION 12 COMPLETED.")