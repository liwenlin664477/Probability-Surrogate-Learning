function [Ypred, model] = mlvHogp_s2d(xtr,Ytr,xte,r)
% hogp square 2d

    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        y_dim = size(Ytr{k},2);
        
        kxtr = xtr(1:nSample,:); 
        
        if(r>nSample)
            r = nSample;
        else 
        end   

        size_new = sqrt(y_dim);
        
        ytr = reshape(Ytr{k}',[size_new,size_new,nSample]);
        ytr = permute(tensor(ytr),[3,1,2]);
        
        ytr = tensor(ytr);
        
        model{k} = hogp_v2(kxtr, ytr, xte, r);
        Ypred{k} = model{k}.yPred.data;
        
        Ypred{k} = reshape(Ypred{k},size(Ypred{k},1),[]);
    end
    
end