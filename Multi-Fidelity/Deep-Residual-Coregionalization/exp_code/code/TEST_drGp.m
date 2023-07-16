% TEST_drGp
clear
rng(1234)
% addpath(genpath('./../../code'))
nLv = 3;

% nTr_lv2_list = 10:20:100;
r = 3; 
nTr{3} = 20;
nTr{2} = nTr{3}*2;
nTr{1} = nTr{2}*2;

dataName = 'syn_v02';
load(dataName)

% r = 6;
%shuffle data
id_rand = randperm(512);
id_tr = id_rand(1:480);
id_te = id_rand(480+1:512);

xtr = X(id_tr, :);
xte = X(id_te, :);

for k = 1:nLv
    Ytr{k} = Y{k}(id_tr(1:nTr{k}),:);
end
yte_golden = Y{nLv+1}(id_te,:);
%%
k=1;
for i = 1:length(Ytr)
    error_true(k) = err(Y{i}(id_te,:),yte_golden);  
    k=k+1;
end
%%
k = 1;

% [Ypred, model] = mlvPcaGp(xtr,Ytr,xte,r);
% for i = 1:length(Ytr)
%     error(k) = err(Ypred{i},yte_golden);  
%     k=k+1;
% end
% 
% [Ypred, model] = mlvKpcaGp(xtr,Ytr,xte,r);
% for i = 1:length(Ytr)
%     error(k) = err(Ypred{i},yte_golden);  
%     k=k+1;
% end
% 
% [Ypred, model] = mlvIsomapGp(xtr,Ytr,xte,r);
% for i = 1:length(Ytr)
%     error(k) = err(Ypred{i},yte_golden);  
%     k=k+1;
% end

[Ypred, model] = mlvHogp(xtr,Ytr,xte,r);
for i = 1:length(Ytr)
    error(k) = err(Ypred{i},yte_golden);  
    k=k+1;
end
%%
function error = err(yPred,yTrue)

    err2 = mean((yPred - yTrue).^2, 2);
    error.mse = mean(err2);
    error.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    error.re = mean(re);
    error.re_std = std(re);
    
    ymean = mean(yTrue(:));
    ystd = std(yTrue(:));
    yTrue = (yTrue - ymean) ./ ystd;
    yPred = (yPred - ymean) ./ ystd;
    
    err2_normal = mean((yPred - yTrue).^2, 2);
    error.nmse = mean(err2_normal);
    error.nmse_std = std(err2_normal);

end
