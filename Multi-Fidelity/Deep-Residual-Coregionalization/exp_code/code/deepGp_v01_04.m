function [Ypred, model] = deepGp_v01_04(xtr,Ytr,xte, iMethod)
% v01: simple deep GP.
% v01_02: same model as v01 but using GP with normalize y x (seperate
%         dimension)
% v01_03: use different normalization y(:) x(:). !BEST! cmp with previous
%         version
% v01_04: introduce cigp_cat training

coreGp_func = @cigp_v2_03;

assert(iscell(Ytr),'Y must be cell');
nLv = length(Ytr);

switch iMethod
    case 1 % independent GP
        for k = 1:nLv
            nSample = size(Ytr{k}, 1);
            kxtr = xtr(1:nSample,:);   
            try
                model{k} = coreGp_func(kxtr, Ytr{k}, xte);
                Ypred{k} = model{k}.yTe_pred;
            catch
                Ypred{k} = zeros(size(xte,1), size(Ytr{k},2));
            end
        end
        
    case 2 % simple concatinate structure
        nSample = size(Ytr{1}, 1);
        kxtr = xtr(1:nSample,:);  
        model{1} = coreGp_func(kxtr, Ytr{1}, xte);
        Ypred{1} = model{1}.yTe_pred;
        for k = 2:nLv
%             model{k} = coreGp_func(kxtr, Ytr{k}, xte);nSample = size(Y{k}, 1);
            nSample = size(Ytr{k}, 1);
            kxte = model{k-1}.yTe_pred;
            
            kxtr = Ytr{k-1}(1:nSample,:);
            try 
                model{k} = coreGp_func(kxtr, Ytr{k}, kxte);
                Ypred{k} = model{k}.yTe_pred;
            catch 
                Ypred{k} = zeros(size(xte,1), size(Ytr{k},2));
            end
        end
    
    case 3 % residual deep GP. case 2 plus x
        nSample = size(Ytr{1}, 1);
        kxtr = xtr(1:nSample,:);  
        model{1} = coreGp_func(kxtr, Ytr{1}, xte);        
%         model{1} = coreGp_func(xtr, Ytr{1}, xte);
        Ypred{1} = model{1}.yTe_pred;
        for k = 2:nLv
%             model{k} = coreGp_func(kxtr, Ytr{k}, xte);nSample = size(Y{k}, 1);
            nSample = size(Ytr{k}, 1);
            
            kxte = [xte, model{k-1}.yTe_pred];    
            kxtr = [xtr(1:nSample,:) ,Ytr{k-1}(1:nSample,:)];
            
            try
                model_temp = coreGp_func(xtr(1:nSample,:), Ytr{k}, xte);
                model{k} = cigp_cat_v03(model_temp, kxtr, Ytr{k}, kxte);
                Ypred{k} = model{k}.yTe_pred;
            catch 
                Ypred{k} = zeros(size(xte,1), size(Ytr{k},2));
            end
                
        end
    otherwise 
        error('no method')
        
end

model_info = model;
model = [];
model.gpModel = model_info;


end