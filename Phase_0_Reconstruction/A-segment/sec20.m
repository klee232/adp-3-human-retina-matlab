function [p2, II, DD] = sec20(uid, eid, did, p1suff, p2suff, opt, pathdata0, pathrepo0)

disp("SECTION 20 RUNNING ...");

%% p2: struct to save all processing parameters

% basic
p2 = struct;
p2.uid = uid;
p2.eid = eid;
p2.did = did;
p2.p1suff = p1suff;
p2.p2suff = p2suff;
p2.id = sprintf("%s-%s-%s", did, p1suff, p2suff);

% folders
%   Similar policy as sec10
p2.pathdata = sprintf("%s/%s", pathdata0, eid);  
p2.pathrepo = sprintf("%s/2-enhance #%s #%s", pathrepo0, p2.eid, p2.id);

% opt
p2.opt = opt;  % processing options given by the user (identified by p1suff)

% print p2 for check
p2
p2.opt


%% Check if p2id file exists

fpath = sprintf("%s/%s.mat", p2.pathdata, p2.id);
if ~isempty(dir(fpath))
    error("Processed data already exists. Rename p2suff or delete the existing file. It is not recommended to rename the existing file:\n%s", ...
        fpath);
end


%% Load p1, r1, II, DD

fpath = sprintf("%s/%s-%s.mat", p2.pathdata, p2.did, p2.p1suff);
l = load(fpath, "p1", "r1", "II", "DD");
p2.p1 = l.p1;
p2.r1 = l.r1;
II = l.II;
DD = l.DD;


disp("SECTION 20 COMPLETED.");
