%% Batch run app1_reconstruct

clear;  close all;  clc;
comInit;

% options
p1suff = "p1a";
bOverwrite = true;

% read meta1a.tsv, copied from gsheet ADP-3 data / OCTA-URI
tbMeta = comReadMeta()

% Run app1 for left eye and right eye's OCTA
DispProg(0, size(tbMeta,1), "BATCH1 ...")
for ie=1:size(tbMeta,1)
% for ie=38:size(tbMeta,1)  % ie=20 & 27 have missing vol file
    if ie == 20 || ie == 27
        continue;
    end
    
    tbMeta1 = tbMeta(ie,:);
    uid1 = tbMeta1.uid;
    eid1 = tbMeta1.eid;
    did1 = split(tbMeta1.did, ", ");

    for id=1:2  % left, right OCTA
        did2 = did1(id);
        if did2 == ""
            warning("No OCTA data: %s | %s", uid1, eid1);
        else
            % check p1 file
            fpath = sprintf("%s/%s/%s-%s.mat", ...
                pathdata0, eid1, did2, p1suff);
            if ~isempty(dir(fpath)) && ~bOverwrite
                warning("p1 file already exists:\n%s", fpath);
            else
                % run app1
                [p1, r1] = app1_reconstruct(uid1, eid1, did2, p1suff);
                close all;

                % update tbMeta.p1id
                if tbMeta.p1id(ie) ~= ""
                    tbMeta.p1id(ie) = sprintf("%s, ", tbMeta.p1id(ie));
                end
                tbMeta.p1id(ie) = sprintf("%s%s", tbMeta.p1id(ie), p1.id);
            end
        end
    end

    DispProg(ie, size(tbMeta,1), "#####################");
end
tbMeta
    
