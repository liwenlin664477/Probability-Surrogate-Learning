function [Ypred, model] = mfrPca_dgp_v0(xtr,Ytr,xte,r,iMethod_dgp)

    dgp_func = @deepGp_v01;
%     dgp_func = @deepGp_v01_02;    dgp with normalize y
%     dgp_func = @deepGp_v01_03;    dgp with normalize y(:)
    
    [Ztr,model_mfrPCA] = mfrPCA(Ytr,r);
    [Zpred, model_dgp] = dgp_func(xtr,Ztr,xte, iMethod_dgp);
    Ypred = model_mfrPCA.recover(Zpred);

    model.model_dgp = model_dgp;
    model.model_mfrPCA = model_mfrPCA;
end