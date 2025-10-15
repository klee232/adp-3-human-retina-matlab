% function [p1, r1] = sec12(p1, r1, II1, DD1)
disp("SECTION 12 RUNNING ...")

[nz, nx, ny] = size(II1);
II2 = imgaussfilt3(II1, [1 1 1]);  % just for figure
DD2 = imgaussfilt3(DD1, [1 1 1]);  % just for figure

%%%% Check p1.hdrBscan.SEG

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
diffSeg = abs(seg(:,:,1) - ilm);
diffSeg = mean(diffSeg(~isnan(diffSeg)));
if diffSeg < 0.01
    fprintf("SEG(:,:,1) = ILM: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,1) ~= ILM: diff = %.2f vox", diffSeg);
end
diffSeg = abs(seg(:,:,2) - rpe);
diffSeg = mean(diffSeg(~isnan(diffSeg)));
if diffSeg < 0.01
    fprintf("SEG(:,:,2) = RPE: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,2) ~= RPE: diff = %.2f vox", diffSeg);
end
diffSeg = abs(seg(:,:,3) - nfl);
diffSeg = mean(diffSeg(~isnan(diffSeg)));
if diffSeg < 0.01
    fprintf("SEG(:,:,3) = NFL: diff = %.2f vox\n", diffSeg);
else
    error("SEG(:,:,3) ~= NFL: diff = %.2f vox", diffSeg);
end

%%%% transpose & nz-

% seg should be flipped by "nz-".
seg = nz - seg;
ilm = nz - ilm;
rpe = nz - rpe;
nfl = nz - nfl;

% transpose
%   Heidelberg's code in ReadVOL_22URI() assign seg(iy,:) for each y
r1.seg = seg;
for is=1:size(r1.seg,3)
    seg1 = r1.seg(:,:,is)';
    seg1(isnan(seg1)) = mean(seg1(~isnan(seg1)));
    r1.seg(:,:,is) = seg1;
end
r1.ilm = ilm';  
r1.rpe = rpe';
r1.nfl = nfl';

%%%% plot r1.seg

y = round(ny * [.25 .5 .75]);
limZ = [min(r1.seg, [], 'all'), max(r1.seg, [], 'all')] + [-10 30];

for im=1:2  % II2 and DD2

    % fid
    fid = split(p1.pathrepo, "/");  fid = fid(end);  
    if im == 1
        ftag = "#12-seg";
    else
        ftag = "#12-seg-ang";
    end    
    fid = sprintf("%s %s", fid, ftag);
    
    clr = lines;
    fig = NewFig2(4.5,5);  colormap(gray);
    for iy=1:3
        y1 = y(iy);
        subplot(1,3,iy);
        if im == 1
            img = II2(:,:,y1);
            imagesc(log(max(img, 1e-3)));
        else
            img = mean(DD2(:,:,y1+(-2:2)), 3);
            imagesc(imadjust(img));
        end
        % axis image;
        for is=1:size(r1.seg,3)
            seg1 = r1.seg(:,:,is);
            if is <= 3  % ILM, RPE, NFL
                lw = 1.5;
            else 
                lw = 0.5;  % default of line()
            end
            line(1:nx, seg1(:,y1), Color=clr(is,:), LineWidth=lw);
            text(nx, seg1(nx,y1), sprintf(" %d",is), Color=clr(is,:));
        end
        ax = gca;
        ax.YLim = limZ;
        ax.Title.String = sprintf("y=%d/%d", y1, ny);
    end
    sgtitle(fid);
    SaveFig(fig, false, fid, sprintf("%s %s", p1.pathrepo, ftag));
end

%%%% r1.lyr
%   3d array with 1 = (layer above ILM), and so on

% seg NaN
segZ = squeeze(mean(mean(r1.seg, 1), 2));
b = isnan(segZ) | segZ == 0;  % 2407a: bad data had 0 instead of NaN
if any(b)
    warning("r1.seg has %d NaN in %s. They will be removed.", ...
        sum(b), mat2str(find(b)'));
end

% seg order
[~, isrt] = sort(segZ);
isrt = isrt(~ismember(isrt, find(b)));
if ~(isrt(1) == 1 && isrt(end) == 2)
    warning("r1.seg(1) should be at the top (ILM) and r1.seg(2) should be at the bottom (RPE), but not:")
end
fprintf("r1.seg order: %s\n", mat2str(isrt'))
seg1 = r1.seg(:,:,isrt);

% lyr
lyr = ones(nz, nx, ny, 'uint8');  % default = 1 (above ILM)
[gz, gx, gy] = ndgrid(1:nz, 1:nx, 1:ny);
for is=1:size(seg1,3)
    seg2 = seg1(:,:,is);
    seg2 = repmat(reshape(seg2, [1 nx ny]), [nz 1 1]);
    lyr(gz >= seg2) = is+1;
end

% zero in the first and last layers
if p1.opt.zAboveIlm > 0
    seg2 = seg1(:,:,1);
    seg2 = repmat(reshape(seg2, [1 nx ny]), [nz 1 1]);
    lyr(gz < seg2 - p1.opt.zAboveIlm) = 0;
end
if p1.opt.zBelowRpe > 0
    seg2 = seg1(:,:,end);
    seg2 = repmat(reshape(seg2, [1 nx ny]), [nz 1 1]);
    lyr(gz > seg2 + p1.opt.zBelowRpe) = 0;
end

% plot
% cmap = jet(size(seg1,3)+1);
cmap = lines(size(seg1,3)+1);
if min(lyr(:)) == 0
    cmap = [0 0 0; cmap];
end
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-layer", fid);
fig = NewFig2(4.5,5);  colormap(cmap);
for iy=1:3
    y1 = y(iy);
    subplot(1,3,iy);
    img = lyr(:,:,y1);
    imagesc(img);
    ax = gca;
    ax.YLim = limZ;
    ax.Title.String = sprintf("y=%d/%d", y1, ny);
    % colorbar;
end
sgtitle(fid);
SaveFig(fig, false, fid, sprintf("%s #12-layer", p1.pathrepo));

r1.lyr = lyr;
r1.nlyr = size(seg1,3)+1;

%%%% en face DD2 for layer

fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-ang-layer", fid);

nmtg = min(r1.nlyr, 15);
mtg = zeros(nx, ny, nmtg, 'single');
for il=1:nmtg
    DD21 = DD2 .* (r1.lyr == il);
    img = squeeze(max(DD21, [], 1));
    % img = squeeze(mean(DD21, 1));
    img = imadjust(img);
    mtg(:,:,il) = flipud(img');
end

fig = NewFig2(4,4);  
montage(mtg, Size=[3 5], BorderSize=[1 1], BackgroundColor='w');
ax = gca;
% ax.XLabel.String = sprintf("r1.nlyr=%d", r1.nlyr);
ax.Title.String = fid;
SaveFig(fig, false, fid, sprintf("%s #12-ang-layer", p1.pathrepo));

%%%% en face DD2 for layer combinations

lyrComb = p1.opt.lyrComb;
lyrComb(lyrComb==Inf) = r1.nlyr;

fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-ang-layer-comb", fid);
mtg = zeros(nx, ny, size(lyrComb,1), 'single');
for il=1:size(lyrComb,1)
    lyrComb1 = lyrComb(il,:);
    DD21 = DD2 .* (r1.lyr >= lyrComb1(1)) .* (r1.lyr <= lyrComb1(2));
    img = squeeze(max(DD21, [], 1));
    % img = squeeze(mean(DD3, 1));
    img = imadjust(img);
    mtg(:,:,il) = flipud(img');
end

fig = NewFig2(2,4);  
montage(mtg, Size=[1 size(mtg,3)], BorderSize=[1 1], BackgroundColor='w');
ax = gca;
% ax.XLabel.String = sprintf("layer combination = %s", mat2str(lyrComb));
ax.Title.String = fid;
SaveFig(fig, false, fid, sprintf("%s #12-ang-layer-comb", p1.pathrepo));

disp("SECTION 12 COMPLETED.")