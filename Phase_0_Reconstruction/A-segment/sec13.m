% function [p1, r1] = sec13(p1, r1, II1, DD1)
disp("SECTION 13 RUNNING ...");

[nz, nx, ny] = size(II1);

%% Mean intensity, decorrelation, Z

r1.lyrI = zeros(r1.nlyr, 1);
r1.lyrD = zeros(r1.nlyr, 1);
r1.lyrZ = zeros(r1.nlyr, 1);
[gz, gx, gy] = ndgrid(1:nz, 1:nx, 1:ny);
for il=1:r1.nlyr
    msk = r1.lyr == il;
    r1.lyrI(il) = mean(II1(msk));
    r1.lyrD(il) = mean(DD1(msk));
    r1.lyrZ(il) = mean(gz(msk));
end

% plot
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #13-bar", fid);
fig = NewFig2(2,3);
bar(1:r1.nlyr, r1.lyrI);
hold on;
bar(1:r1.nlyr, -r1.lyrD);
ax = gca;
ax.XLabel.String = "Layer index";
ax.YLabel.String = ["Mean intensity (blue)", "Mean decorrelation (red)"];
ax.Title.String = fid;
SaveFig(fig, false, fid, sprintf("%s #13-bar", p1.pathrepo));


%% Scatter

zRel = r1.lyrZ - r1.lyrZ(2);  % relative distance

fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #13-scatter", fid);
fig = NewFig2(3,3);  colormap(jet);

subplot(211);
scatter(r1.lyrI, r1.lyrD, [], zRel, 'filled', Marker='o');
axis image;
grid on;
ax = gca;
ax.XLabel.String = ["OCT (intensity)", "color=depth (red=deeper)"];
ax.YLabel.String = "OCTA (decorrelation)";
ax.Title.String = fid;

subplot(212);
scatter3(r1.lyrI, r1.lyrD, -zRel, [], zRel, "filled", Marker='o');
ax = gca;
ax.DataAspectRatio(2) = ax.DataAspectRatio(1);
ax.XLabel.String = "OCT";
ax.YLabel.String = "OCTA";
ax.ZLabel.String = "Depth (vox)";

SaveFig(fig, false, fid, sprintf("%s #13-scatter", p1.pathrepo));


disp("SECTION 13 COMPLETED.");