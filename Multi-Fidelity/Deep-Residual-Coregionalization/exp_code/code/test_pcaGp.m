% test_pcaGp

clear
rng(1234)
% addpath(genpath('./../../code'))
dataName = 'syn_v02';
load(dataName)

ytr = Y{4}(1:96,:);
yte = Y{4}(257:512,:);
xtr = X(1:96,:);
xte = X(257:512,:);

%also try different normalization method
gp_func = @cigp_v2;
gp_func = @cigp_v2_02;
gp_func = @cigp_v2_03;


%try both the following
r_list = 2:2:16;
r_list = 10;
% r_list = [0.8,0.9,0.99,0.999,0.9999,0.99999]
%%
for ir = 1:length(r_list)
    r = r_list(ir);
    
    [ztr,model_pca] = pcaInit(ytr,r);
    model_gp = gp_func(xtr, ztr, xte);
    yPred = model_pca.recover(model_gp.yTe_pred);
    
    err(ir,1) = mse(yPred,yte);
    
    [ztr,model_pca] = pcaInit_v02(ytr,r);
    model_gp = gp_func(xtr, ztr, xte);
    yPred = model_pca.recover(model_gp.yTe_pred);
    
    err(ir,2) = mse(yPred,yte);
    
    [ztr,model_pca] = pcaInit_v03(ytr,r);
    model_gp = gp_func(xtr, ztr, xte);
    yPred = model_pca.recover(model_gp.yTe_pred);
    
    err(ir,3) = mse(yPred,yte);
    
end

%%
plot(r_list,log(err),'o--')
legend('1','2','3')








%%

function err = mse(yPred,ytrue)
    err = mean2((yPred-ytrue).^2);
end

function [yPred,model] = pcaGp(xtr,ytr,xte,r)

gp_func = @cigp_v2_03;

[ztr,model_pca] = pcaInit_v02(ytr,r);

model_gp = gp_func(xtr, ztr, xte);
yPred = model_pca.recover(model_gp.yTe_pred);


model.model_gp = model_gp;
model.model_pca = model_pca;
end