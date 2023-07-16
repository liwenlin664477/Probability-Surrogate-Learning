function [Ypred, model] = deepGp_v01(xtr,Ytr,xte, iMethod)
% v01: simple deep GP.
assert(iscell(Ytr),'Y must be cell');
nLv = length(Ytr);

switch iMethod
    case 1 % independent GP
        for k = 1:nLv
            nSample = size(Ytr{k}, 1);
            kxtr = xtr(1:nSample,:);            
            model{k} = cigp_v2(kxtr, Ytr{k}, xte);
            Ypred{k} = model{k}.yTe_pred;
        end
        
    case 2 % simple concatinate structure
        nSample = size(Ytr{1}, 1);
        kxtr = xtr(1:nSample,:);  
        model{1} = cigp_v2(kxtr, Ytr{1}, xte);
        Ypred{1} = model{1}.yTe_pred;
        for k = 2:nLv
%             model{k} = cigp_v2(kxtr, Ytr{k}, xte);nSample = size(Y{k}, 1);
            nSample = size(Ytr{k}, 1);
            kxte = model{k-1}.yTe_pred;
            
            kxtr = Ytr{k-1}(1:nSample,:);
            model{k} = cigp_v2(kxtr, Ytr{k}, kxte);
            Ypred{k} = model{k}.yTe_pred;
        end
    
    case 3 % residual deep GP. case 2 plus x
        nSample = size(Ytr{1}, 1);
        kxtr = xtr(1:nSample,:);  
        model{1} = cigp_v2(kxtr, Ytr{1}, xte);        
%         model{1} = cigp_v2(xtr, Ytr{1}, xte);
        Ypred{1} = model{1}.yTe_pred;
        for k = 2:nLv
%             model{k} = cigp_v2(kxtr, Ytr{k}, xte);nSample = size(Y{k}, 1);
            nSample = size(Ytr{k}, 1);
            kxte = [xte, model{k-1}.yTe_pred];    
            kxtr = [xtr(1:nSample,:) ,Ytr{k-1}(1:nSample,:)];
            model{k} = cigp_v2(kxtr, Ytr{k}, kxte);
            Ypred{k} = model{k}.yTe_pred;
        end
    otherwise 
        error('no method')
        
end

model_info = model;
model = [];
model.gpModel = model_info;


end