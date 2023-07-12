function [Ypred, model] = mlvHogp_v01(xtr,Ytr,xte,r)

%     nlv = length(Ytr);
%     Ypred = cell(1,nlv);
%     model = cell(1,nlv);
    
    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:); 
        
        if(r>nSample)
            r = nSample;
        else 
        end    
        
        try
            ytr = reshape(Ytr{k}',[100,100,nSample]);
            ytr = permute(tensor(ytr),[3,1,2]);

            ytr = tensor(ytr);

            model{k} = hogp_v2(kxtr, ytr, xte, r);
            Ypred{k} = model{k}.yPred.data;

            Ypred{k} = reshape(Ypred{k},size(Ypred{k},1),[]);
        catch 
            Ypred{k} = [];
            model{k} = [];
        end
    end
    
end