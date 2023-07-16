function [Ypred, model] = mfrPca_dgpx3(xtr,Ytr,xte,r)

dgp_func = @deepGp_v01;
%     dgp_func = @deepGp_v01_02;    dgp with normalize y
%     dgp_func = @deepGp_v01_03;    dgp with normalize y(:)

for iMethod_dgp = 1:3
    Ypred{iMethod_dgp} = [];

    [Ztr,model_mfrPCA] = mfrPCA(Ytr,r);
    [Zpred, model_dgp] = dgp_func(xtr,Ztr,xte, iMethod_dgp);
    Ypred_i = model_mfrPCA.recover(Zpred);
    
    Ypred{iMethod_dgp} = Ypred_i{end};

    model{iMethod_dgp}.model_dgp = model_dgp;
    model{iMethod_dgp}.model_mfrPCA = model_mfrPCA;
    
end

end