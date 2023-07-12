% TEST_v03
% v02: run multiple exp setting
% v03: test residual deep gp
clear
rng(1234)
% addpath(genpath('./../../code'))
dataName = 'syn_v02';
load(dataName)

Ytr{1} = Y{1}(1:256,:);
Ytr{2} = Y{2}(1:128,:);
Ytr{3} = Y{3}(1:64,:);
Ytr{4} = Y{4}(1:16,:);

Yte{1} = Y{1}(257:512,:);
Yte{2} = Y{2}(257:512,:);
Yte{3} = Y{3}(257:512,:);
Yte{4} = Y{4}(257:512,:);

xtr = X(1:256,:);
xte = X(257:512,:);

%% main 
%residual gp: test different normalizing
r=0.9999;
[Ztr,model_mfrPCA] = mfrPCA(Ytr,r);

k = 0;
for iMethod = 1:3
    k=k+1;
    [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod);
    Ypred = model_mfrPCA.recover(Zpred);
    err(k) = err_eval(Ypred{end},Yte{end});
end

for iMethod = 1:3
    k=k+1;
    [Zpred, model] = deepGp_v01_02(xtr,Ztr,xte, iMethod);
    Ypred = model_mfrPCA.recover(Zpred);
    err(k) = err_eval(Ypred{end},Yte{end});
end

for iMethod = 1:3
    k=k+1;
    [Zpred, model] = deepGp_v01_03(xtr,Ztr,xte, iMethod);
    Ypred = model_mfrPCA.recover(Zpred);
    err(k) = err_eval(Ypred{end},Yte{end});
end

%% only pca
k=0;
% r=0.999;
for i = 1:length(Ytr)
    k=k+1;
    [ztr,model2] = pcaInit(Ytr{i},r);
    
    ntr = size(ztr,1);
    model = cigp_v2(xtr(1:ntr,:),ztr,xte);
    ypred = model2.recover(model.yTe_pred);
    err2(k) = err_eval(ypred,Yte{end});
end

for i = 1:length(Ytr)
    k=k+1;
    [ztr,model2] = pcaInit(Ytr{i},r);
    
    ntr = size(ztr,1);
    model = cigp_v2_02(xtr(1:ntr,:),ztr,xte);
    ypred = model2.recover(model.yTe_pred);
    err2(k) = err_eval(ypred,Yte{end});
end

for i = 1:length(Ytr)
    k=k+1;
    [ztr,model2] = pcaInit(Ytr{i},r);
    
    ntr = size(ztr,1);
    model = cigp_v2_03(xtr(1:ntr,:),ztr,xte);
    ypred = model2.recover(model.yTe_pred);
    err2(k) = err_eval(ypred,Yte{end});
end
%%
function err = err_eval(yPred,yTrue)

    if isempty(yPred)
        yPred = nan(size(yTrue));
    end
             
    err2 = mean((yPred - yTrue).^2, 2);   
    err.mse_sample = err2;
    err.mse_dims = mean((yPred - yTrue).^2, 1); 
    
    err.mse = mean(err2);
    err.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    err.re = mean(re);
    err.re_std = std(re);
    
    yMean = mean(yTrue(:));
    yStd = std(yTrue(:));
    
    %normalize error
    yPred = (yPred - yMean)/yStd;
    yTrue = (yTrue - yMean)/yStd;
    
    nerr2 = mean((yPred - yTrue).^2, 2);
    err.nmse = mean(nerr2);
    err.nmse_std = std(nerr2);

end