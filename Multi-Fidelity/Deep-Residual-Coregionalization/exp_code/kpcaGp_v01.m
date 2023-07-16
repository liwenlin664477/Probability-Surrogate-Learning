function [yPred, model] = kpcaGp_v01(xtr,ytr,xte,r)
% use normal y(:)

  options.ker='gaussian';   
    options.new_dim=r;
    options.FullRec=0;       
    options.arg=0.2;   %10 wont work well,WHY? model.K is too normal distributed which leads to slow eigenvalue decay!!! So options.arg tends to be large.
    options.kAuto=1;    % automatic parameter choosing
    
    preoptions.type='Exp';
    preoptions.neighbor=10;
    
    nSample = size(ytr,1);
    
    if(r>nSample)
        options.new_dim = nSample;
    else 
        options.new_dim = r;
    end

    try
        [ztr,kpcaModel] = kpcaInit(ytr,options);
%         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
        model_gp = cigp_v2_03(xtr, ztr, xte);
        yPred = Kpca_PreImage2(model_gp.yTe_pred,kpcaModel,preoptions);
%         train_pred = Kpca_PreImage2(utrain_pred(:,1:k),kpcaModel,preoptions);
    catch 
        yPred = [];
    end
        
    model.kpcaModel = kpcaModel;
    model.model_gp = model_gp;   
end