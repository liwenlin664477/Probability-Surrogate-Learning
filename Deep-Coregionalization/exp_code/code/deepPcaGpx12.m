function [Ypred, model] = deepPcaGpx12(xtr,Ytr,xte,r)

    k = 1;
    for iMethod_pca = 1:4
        for iMethod_dgp = 1:3     
                try
                    [Ypred_k, model{k}] = deepPcaGp_v01(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
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