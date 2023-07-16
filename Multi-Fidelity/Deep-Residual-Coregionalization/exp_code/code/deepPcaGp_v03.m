
function [Ypred, model] = deepPcaGp_v03(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp)

%     [Ztr,U] = multiLvPca(Ytr, r, iMethod_pca);
% %     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
%     [Zpred, model] = deepGp_v01_02(xtr,Ztr,xte, iMethod_dgp);
%     
%     for k = 1:length(Ytr)
%        Ypred = Zpred{k} * U{k};
%     end
%     
%     model.U = U;
%     model.Ztr = Ztr;
    
    
    [Ztr,U] = multiLvPca(Ytr, r, iMethod_pca);
%     [Ztr,U] = multiLvPca_v2(Ytr, r, iMethod_pca);
%     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
%     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
    [Zpred, model] = deepGp_v01_03(xtr,Ztr,xte, iMethod_dgp);
    
    Ypred = multiLvPcaInv(Zpred, U, iMethod_pca);
%     for k = 1:length(Ytr)
%        Ypred{k} = Zpred{k} * U{k};
%     end
    
    model.U = U;
    model.Ztr = Ztr;
    
    
    
end
