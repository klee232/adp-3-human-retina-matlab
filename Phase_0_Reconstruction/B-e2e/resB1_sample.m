%% Firs trial of end-to-end pipeline

%% simple example codes
clear

% load meta data file
comInit
tbMeta = comReadMeta()

% load image data
%   this is a result of ../A-segment/app1_reconstruct
%   input argument: eid, did, p1suff
%   available p1suff: p1a
%   output:
%       II: OCT image (z, x, y)
%       DD: OCTA image (z, x, y)
%       p1: app1_reconstruct's various parameters
%           p1.hdr: OCT imaging's header information provided by the OCT machine
%           p1.hdrBscan: OCT image analysis's header information provided by the OCT machine
%           p1.hdrAng: OCTA imaging's header information provided by the OCT machine
%           p1.hdrBscanAng: OCTA image anlaysis's header information provided by the OCT machine
%       r1: app1_reconstruct's various results
%           r1.slo: OCT imaging's SLO image (x, y) (SLO: camera-like image)
%           r1.sloAng: OCTA imaging's SLO image (x, y)
%           r1.seg: OCT image's layer boundaries' Z coordinates (x, y, boundary number): similar as above p1.hdrBscan.SEG
%           r1.ilm, rpe, nfl: Z coordinates of the three specific layers (ILM, RPE, NFL) (x, y) for each
%           r1.lyr: Layer label image, contains layer number of each voxel (z, x, y): 0 = space above the retina
[p1, r1, II, DD] = comLoadDataA_p1("101-1003-y1", "473", "p1a");

%% sample code to plot en face image of a few layer combinations
%   this code produces a same figure as ~/data/report/adp-3-human-retina/A-segment/<eid>/<did>.png

% option
lyrComb = [2 6; 8 8; 14 14];  % layer combinations to plot: first 2-6 (surface), second 8-8 (deep capillary), third 14-14 (choroid)

% image processing
[nz, nx, ny] = size(II);
II1 = imgaussfilt3(II, [1 1 1]);  
DD1 = imgaussfilt3(DD, [1 1 1]);  

% montage
mtg = zeros(nx, ny, size(lyrComb,1), 'single');
for il=1:size(lyrComb,1)
    lyrComb1 = lyrComb(il,:);
    DD2 = DD1 .* (r1.lyr >= lyrComb1(1)) .* (r1.lyr <= lyrComb1(2));
    img = squeeze(max(DD2, [], 1));
    img = imadjust(img);
    mtg(:,:,il) = flipud(img');
end

% plot
fid = split(p1.pathrepo, "/");  fid = fid(end);  
fid = sprintf("%s #12-ang-layer-comb", fid);
fig = NewFig2(2,4);  
montage(mtg, Size=[1 size(mtg,3)], BorderSize=[1 1], BackgroundColor='w');
ax = gca;
% ax.XLabel.String = sprintf("layer combination = %s", mat2str(lyrComb));
ax.Title.String = fid;

% % save the processed 3D image file
% OCTA_dir_name="processed_img\OCTA\";
% if ~isfolder(OCTA_dir_name)
%     mkdir(OCTA_dir_name)
% end
% OCT_dir_name="processed_img\OCT\";
% if ~isfolder(OCT_dir_name)
%     mkdir(OCT_dir_name)
% end

% file_name_OCTA=strcat(p1.uid,"-",p1.eid,"-",p1.id,"-OCTA",".mat");
% file_name_OCT=strcat(p1.uid,"-",p1.eid,"-",p1.id,"-OCT",".mat");
% 
% save_dir_OCTA=strcat(OCTA_dir_name,file_name_OCTA);
% save_dir_OCT=strcat(OCT_dir_name,file_name_OCT);
% 
% save(save_dir_OCTA,"DD1")
% save(save_dir_OCT,"II1")

