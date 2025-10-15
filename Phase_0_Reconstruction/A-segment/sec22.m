function [p2, II, DD] = sec21(p2, II0, DD0)

disp("SECTION 21 RUNNING ...");

medfsize = p2.opt.medfsize;
gaufstd = p2.opt.gaufstd;


%% Filter

II = II0;  DD = DD0;

% median
if sum(medfsize) > 0
    II = medfilt3(II, medfsize);
    DD = medfilt3(DD, medfsize);
end

% Gaussian
if sum(gaufstd) > 0
    II = imgaussfilt3(II, gaufstd);
    DD = imgaussfilt3(DD, gaufstd);
end



%% En face by layer


disp("SECTION 21 COMPLETED.")