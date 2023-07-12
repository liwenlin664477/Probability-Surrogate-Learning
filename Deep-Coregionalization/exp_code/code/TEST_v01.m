% TEST_v01
clear
rng(1234)
% addpath(genpath('./../../code'))
nLv = 3;

% nTr_lv2_list = 10:20:100;
r = 20; 
nTr{3} = 60;
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
% yte_golden = Y{nLv}(id_te,:);

%%
% yte1 = MFFGP_ub(r,xtr,ytr,xte);

%% 
% [Ztr,Utr] = multiLvPca(ytr, 5, 1);
% [Zpred, model] = deepGp_v01(xtr,Ztr,xte, 3);

%%

k=1;
for iMethod_pca = 1:4
    for iMethod_dgp = 1:3     
        [Ypred, model] = deepPcaGp_v01(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
        error(k) = err(Ypred{nLv},yte_golden);  
        k=k+1;
        [Ypred, model] = deepPcaGp_v03(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
        error(k) = err(Ypred{nLv},yte_golden);  
        k=k+1;
    end
end

%%
[Ypred, model] = pcaGp(xtr,Ytr,xte,r);

error(k) = err(Ypred{1},yte_golden);  
k=k+1;
error(k) = err(Ypred{2},yte_golden);  
k=k+1;
error(k) = err(Ypred{3},yte_golden);  
k=k+1;

%%
function [Ypred, model] = deepPcaGp_v01(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp)

    [Ztr,U] = multiLvPca(Ytr, r, iMethod_pca);
%     [Ztr,U] = multiLvPca_v2(Ytr, r, iMethod_pca);
%     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
%     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
    [Zpred, model] = deepGp_v01_02(xtr,Ztr,xte, iMethod_dgp);
    
    Ypred = multiLvPcaInv(Zpred, U, iMethod_pca);
%     for k = 1:length(Ytr)
%        Ypred{k} = Zpred{k} * U{k};
%     end
    
    model.U = U;
    model.Ztr = Ztr;
end


function [Ypred, model] = pcaGp(xtr,Ytr,xte,r)

    for k = 1:length(Ytr)
        [U{k}, Ztr{k}] = myPca(Ytr{k}, r);
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        model{k} = cigp_v2_02(kxtr, Ztr{k}, xte);
    
        Ypred{k} = model{k}.yTe_pred * U{k};
    end

end

function error = err(yPred,yTrue)

    err2 = mean((yPred - yTrue).^2, 2);
    error.mse = mean(err2);
    error.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    error.re = mean(re);
    error.re_std = std(re);

end

