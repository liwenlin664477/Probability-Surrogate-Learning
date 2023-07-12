function [Ypred, model] = deepPcaGpx12_v02(xtr,Ytr,xte,r)
% v02 use different normalization y(:) x(:)

    k = 1;
    for iMethod_pca = 1:4
        for iMethod_dgp = 1:3     
                try
                    [Ypred_k, model{k}] = deepPcaGp_v03(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
%                     errRec = errRec_write(errRec, Ypred, yte, i,j,k);
%                     err(k) = err_eval(Ypred{k},yte);
                    Ypred{k} = Ypred_k{end};
                catch
                    Ypred{k}=[]; model{k}=[];
                end
                k=k+1;
            end
        end
    
end