% function sec19(p1, r1, II, DD)
disp("SECTION 19 RUNNING ...")

% print p1, r1
p1
r1

fpath = sprintf("%s/%s.mat", p1.pathdata, p1.id);
save(fpath, "p1", "r1", "II1", "DD1", '-v7.3');
fprintf("Saved p1, r1, II1, DD1 to:\n%s\n", fpath);

disp("SECTION 19 COMPLETED.")