% TEST_multiLvPca

clear
rng(1234)
% addpath(genpath('./../../code'))
nLv = 3;

% nTr_lv2_list = 10:20:100;
r = 5; 
nTr{3} = 10;
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
for iMethod_pca = 1:4
    [Ztr,U] = multiLvPca(Ytr, r, iMethod_pca);
    Ypred = multiLvPcaInv(Ztr, U, iMethod_pca);
    error(k) = err(Ypred{nLv},Ytr{nLv});  
    k=k+1;
end

for iMethod_pca = 1:4
    [Ztr,U] = multiLvPca_v2(Ytr, r, iMethod_pca);
    Ypred = multiLvPcaInv(Ztr, U, iMethod_pca);
    error(k) = err(Ypred{nLv},Ytr{nLv});  
    k=k+1;
end


%%
function error = err(yPred,yTrue)

    err2 = mean((yPred - yTrue).^2, 2);
    error.mse = mean(err2);
    error.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    error.msre = mean(re);
    error.msre_std = std(re);

end
