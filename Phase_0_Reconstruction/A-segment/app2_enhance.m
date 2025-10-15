%% Enhance OCTA images of human retina
%   Output: 

function app2_enhance(uid, eid, did, p1suff, p2suff)

%% 20. Init

% comment out this when using app1 is called as a function
%
close all;  clear;  clc;
uid = "jonghwan";
eid = "101-1017-y1";
did = "555";
p1suff = "p1a";  % suffix of p1id to load
p2suff = "p2a";  % suffix of p2id (Process 2 ID)
%

comInit;

% parallel
%   we may not parallel for this process of Heidenberg data
% SetParPool(8);  

% opt
opt = struct;
switch p2suff
    case "p2a"
        % default opt
        opt.bZeroAboveIlm = true;
            % Whether to set all voxels above ILM to zero.
        opt.bZeroBelowRpe = true;
            % Whether to set all voxels below RPE to zero.
        opt.bKeepBelowRpe = false;
            % Whether to keep voxels below RPE (as thick as the retina).
            % This has meaning only when bZeroBelowRpe = false.
        opt.zCropMargin = 5;
            % Marginal voxel number in Z when cropping. We will crop out z
            % < min(ILM) - <this> and z > max(RPE) + <this> (when
            % bKeepBelowRpe = false).
        opt.medfsize = [3 3 3];
            % Kernel size of median filtering before averaging. 
            % No when [0 0 0]
        opt.gaufstd = [1 1 1]; 
            % Standard deviation for Gaussian filtering after volume average. 
            % No when [0 0 0]. 
        opt.layBnd = [1 3 2];
            % Retinal layers as the index of p2.r1.seg. En face images will
            % be presented by each layer, as defined by
            % layBnd(1):layBnd(2), layBnd(2):layBnd(3), etc.

    case "p2b"
        % p2a but no median and Gaussian filtering: any effect on final
        % result?
        opt.medfsize = [0 0 0];
        opt.gaufstd = [0 0 0];
        opt.layBnd = [1 3 2];

    otherwise
        error("Not supported p2suff: %s", p2suff);
end

[p2, II0, DD0] = sec20(uid, eid, did, p1suff, p2suff, opt, pathdata0, pathrepo0);  
clear uid eid did p1suff p2suff opt pathraw0 pathdata0 pathrepo0 pathtemp0;


%% 21. Crop over Z
%   After cropping, r1.seg/ilm/rpe/nfl has been modified by transpose and 
%   "nz-", so we don't need to use transpose and nz-ilm anymore. 
%   This section temporarily applies Gaussian filtering (std=1) to II and
%   DD for report figures. The output II2 and DD2 are not affected by this.


%% 22. Filter OCT and OCTA images

[p2, II1, DD1] = sec21(p2, II0, DD0);


%% 12. Crop over Z
%   After cropping, r1.seg/ilm/rpe/nfl has been applied "nz-", so we don't
%   need to use nz-ilm anymore. 
%   This section temporarily applies Gaussian filtering (std=1) to II and
%   DD for report figures. The output II2 and DD2 are not affected by this.

[p1, r1, II2, DD2] = sec12(p1, r1, II1, DD1);


%% 19. Save

sec19(p1, r1, II2, DD2);

