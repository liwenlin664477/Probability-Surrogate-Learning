function [yPred,model] = pcaGp_v02(xtr,ytr,xte,r)

gp_func = @cigp_v2_03;

[ztr,model_pca] = pcaInit_v03(ytr,r);

model_gp = gp_func(xtr, ztr, xte);
yPred = model_pca.recover(model_gp.yTe_pred);


model.model_gp = model_gp;
model.model_pca = model_pca;
end