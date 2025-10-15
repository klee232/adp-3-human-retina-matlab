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

data_list=["101-1017-y1","555", "p1a";... % 1017 left
           "101-1017-y1","550", "p1a";... % 1017 right
           "101-1032-y1","602", "p1a";... % 1032 left
           "101-1032-y1","597", "p1a";... % 1032 right
           "101-1121-y1","673", "p1a";... % 1121 left
           "101-1121-y1","669", "p1a";... % 1121 right
           "101-1149-y1","729", "p1a";... % 1149 right
           "101-1055-y1","612", "p1a";... % 1055 left
           "101-1055-y1","607", "p1a";... % 1055 right
           "101-1136-y1","707", "p1a";... % 1136 left
           "101-1136-y1","702", "p1a";... % 1136 right
           "101-1142-y1","715", "p1a";... % 1142 left
           "101-1142-y1","716", "p1a";... % 1142 right
           "101-1157-y1","738", "p1a";... % 1157 left
           "101-1157-y1","733", "p1a";... % 1157 right
           "101-1181-y1","754", "p1a";... % 1181 left
           "101-1181-y1","749", "p1a";... % 1181 right
           "101-1188-y1","764", "p1a";... % 1188 left
           "101-1188-y1","759", "p1a";... % 1188 right
           "101-1212-y1","784", "p1a";... % 1212 left
           "101-1212-y1","779", "p1a";... % 1212 right
           "101-1260-y1","824", "p1a";... % 1260 left
           "101-1260-y1","819", "p1a";... % 1260 right
           "101-1123-y1","679", "p1a";... % 1123 right
           "101-1127-y1","687", "p1a";... % 1127 left
           "101-1127-y1","595", "p1a";... % 1127 right
           "101-1135-y1","697", "p1a";... % 1135 left
           "101-1135-y1","692", "p1a";... % 1135 right
           "101-1003-y1","473", "p1a";... % 1003 left
           "101-1003-y1","474", "p1a";... % 1003 right
           "101-1004-y1","481", "p1a";... % 1004 left
           "101-1004-y1","482", "p1a";... % 1004 right
           "101-1007-y1","491", "p1a";... % 1007 left
           "101-1007-y1","492", "p1a";... % 1007 right
           "101-1008-y1","506", "p1a";... % 1008 left
           "101-1008-y1","507", "p1a";... % 1008 right
           "101-1009-y1","516", "p1a";... % 1009 right
           "101-1010-y1","526", "p1a";... % 1010 left
           "101-1010-y1","521", "p1a";... % 1010 right
           "101-1011-y1","545", "p1a";... % 1011 left
           "101-1011-y1","540", "p1a";... % 1011 right
           "101-1012-y1","535", "p1a";... % 1012 left
           "101-1012-y1","530", "p1a";... % 1012 right
           "101-1018-y1","564", "p1a";... % 1018 left
           "101-1018-y1","565", "p1a";... % 1018 right
           "101-1019-y1","575", "p1a";... % 1019 left
           "101-1019-y1","570", "p1a";... % 1019 right
           "101-1020-y1","580", "p1a";... % 1020 left
           "101-1056-y1","622", "p1a";... % 1056 left
           "101-1056-y1","617", "p1a";... % 1056 right
           "101-1065-y1","632", "p1a";... % 1065 right
           "101-1068-y1","643", "p1a";... % 1068 left
           "101-1068-y1","638", "p1a";... % 1068 right
           "101-1250-y1","814", "p1a";... % 1250 left
           "101-1250-y1","809", "p1a";... % 1250 right
           "101-1026-y1","590", "p1a";... % 1026 left
           "101-1026-y1","585", "p1a";... % 1026 right
           "101-1006-y1","501", "p1a";... % 1006 right
           "101-1070-y1","653", "p1a";... % 1070 left
           "101-1070-y1","648", "p1a";... % 1070 right
           "101-1071-y1","663", "p1a";... % 1071 left
           "101-1071-y1","658", "p1a";... % 1071 right
           "101-1143-y1","722", "p1a";... % 1143 right
           "101-1203-y1","774", "p1a";... % 1203 left
           "101-1203-y1","769", "p1a";... % 1203 right
           "101-1218-y1","794", "p1a";... % 1218 left
           "101-1218-y1","789", "p1a";... % 1218 right
           "101-1242-y1","804", "p1a";... % 1242 left
           "101-1242-y1","799", "p1a";... % 1242 right
           "101-1280-y1","836", "p1a";... % 1280 left
           "101-1280-y1","831", "p1a";... % 1280 right
           "101-2004-y1","846", "p1a";... % 2004 left
           "101-2004-y1","841", "p1a";... % 2004 right
           "101-2022-y1","865", "p1a";... % 2022 left
           "101-2022-y1","860", "p1a"]; % 2022 right

miss_data_list=[];
unprocess_data_list=[];

%% loop through each data set
[num_data,~]=size(data_list);
for i_data=1:num_data
    % grab out current data
    current_data=data_list(i_data,:);
    current_sample=current_data(1);
    current_did=current_data(2);
    current_process=current_data(3);

    % check if the current file exists
    fpath = sprintf("~/data/procdata/adp-3-human-retina/A-segment/%s/%s-%s.mat", current_sample, current_did, current_process);
   
    % execute the function only when the file exists
    if isfile(fpath)
        [p1, r1, DD, II, indicator] = comLoadDataA_p1(current_sample, current_did, current_process);
        
        if indicator==0
           unprocess_data_list=cat(1,unprocess_data_list,current_data);
           miss_data_list=cat(1,miss_data_list,current_data);
           continue;
        end


        %% sample code to plot en face image of a few layer combinations
        %   this code produces a same figure as ~/data/report/adp-3-human-retina/A-segment/<eid>/<did>.png

        % option
        lyrComb = [2 6; 8 8; 14 14];  % layer combinations to plot: first 2-6 (surface), second 8-8 (deep capillary), third 14-14 (choroid)
        
        % image processing
        [nz, nx, ny] = size(DD);
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
        
        %% save the processed 3D image file
        % OCTA_dir_name="processed_img/OCTA/";
        % if ~isfolder(OCTA_dir_name)
        %     mkdir(OCTA_dir_name)
        % end
        OCT_dir_name="processed_img/OCT/";
        if ~isfolder(OCT_dir_name)
            mkdir(OCT_dir_name)
        end
        
        % file_name_OCTA=strcat(p1.uid,"-",p1.eid,"-",p1.id,"-OCTA",".mat");
        file_name_OCT=strcat(p1.uid,"-",p1.eid,"-",p1.id,"-OCT",".mat");
        % 
        % save_dir_OCTA=strcat(OCTA_dir_name,file_name_OCTA);
        save_dir_OCT=strcat(OCT_dir_name,file_name_OCT);

        % 
        % save(save_dir_OCTA,"DD1","r1")
        save(save_dir_OCT,"II1")
    else
        miss_data_list=cat(1,miss_data_list,current_data);
    end

end

