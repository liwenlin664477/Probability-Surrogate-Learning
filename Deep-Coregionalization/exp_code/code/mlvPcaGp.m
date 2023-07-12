function [yPred, model] = mlvPcaGp(xtr,Ytr,xte,r)

    for k = 1:length(Ytr)        
        
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        
        if(r>nSample)
            r = nSample;
        else 
        end    
        
        [yPred{k},model] = pcaGp(kxtr,Ytr{k},xte,r);
        
%         [U{k}, Ztr{k}] = myPca(Ytr{k}, r);
% %         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
%         model{k} = cigp_v2_02(kxtr, Ztr{k}, xte);
%         
%         Ypred{k} = model{k}.yTe_pred * U{k};


    end

end