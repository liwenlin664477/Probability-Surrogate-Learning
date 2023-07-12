function [Ypred, model] = mfrPca_dgp(xtr,Ytr,xte,r)

% dgp_func = @deepGp_v01;
%     dgp_func = @deepGp_v01_02;   % dgp with normalize y
    dgp_func = @deepGp_v01_04;   % dgp with normalize y(:)
    
%try mfrPCA_v02
for iMethod_dgp = 3
    try 
        [Ztr,model_mfrPCA] = mfrPCA_v02(Ytr,r);
        [Zpred, model_dgp] = dgp_func(xtr,Ztr,xte, iMethod_dgp);
        Ypred_i = model_mfrPCA.recover(Zpred);

        Ypred = Ypred_i{end};

        model.model_dgp = model_dgp;
        model.model_mfrPCA = model_mfrPCA;
    catch 
        Ypred = [];
        model = [];
    end
    
end

end