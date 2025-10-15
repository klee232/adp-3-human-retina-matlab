function tbMeta = comReadMeta()

tbMeta = readtable("../meta-A.tsv", FileType="text", Delimiter='\t');
tbMeta.SubjectID = string(tbMeta.SubjectID);
tbMeta.Group = string(tbMeta.Group);
tbMeta.Sex = string(tbMeta.Sex);
tbMeta.Time = string(tbMeta.Time);
tbMeta.uid = string(tbMeta.uid);
tbMeta.eid = string(tbMeta.eid);

% did
tbMeta.did = string(tbMeta.did_OCTA_l_OCTA_r_OCTv_l_OCTv_r_OCTc_l_OCTc_r_);
tbMeta.did_OCTA_l_OCTA_r_OCTv_l_OCTv_r_OCTc_l_OCTc_r_ = [];

% p1id-p3id
tbMeta.p1id = strings(size(tbMeta,1),1);
tbMeta.p2id = strings(size(tbMeta,1),1);
tbMeta.p3id = strings(size(tbMeta,1),1);

% strong
tbMeta.score = string(tbMeta.score);
tbMeta.score = strrep(tbMeta.score, " ", "");

% note
tbMeta.note = string(tbMeta.Note);
tbMeta.Note = [];

% remove eid=""
tbMeta = tbMeta(tbMeta.eid ~= "", :);