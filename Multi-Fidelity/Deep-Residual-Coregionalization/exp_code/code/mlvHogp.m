function [Ypred, Yvar, beta, model] = mlvHogp(xtr,Ytr,xte,r)

    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:); 
        
        if(r>nSample)
            r = nSample;
        else 
        end    
        
        ytr = reshape(Ytr{k}',[32,32,nSample]);
        ytr = permute(tensor(ytr),[3,1,2]);
        
        ytr = tensor(ytr);
        
        model{k} = hogp_v2(kxtr, ytr, xte, r);
        Ypred{k} = model{k}.yPred.data;
        Ypred{k} = reshape(Ypred{k},size(Ypred{k},1),[]);
        Yvar{k} = model{k}.pred_var.data;
        Yvar{k} = reshape(Yvar{k},size(Yvar{k},1),[]);
        beta{k} = model{k}.bta;
    end
    
end