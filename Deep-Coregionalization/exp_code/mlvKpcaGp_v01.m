function [Ypred, model] = mlvKpcaGp_v01(xtr,Ytr,xte,r)

    options.ker='gaussian';   
    options.new_dim=r;
    options.FullRec=0;       
    options.arg=0.2;   %10 wont work well,WHY? model.K is too normal distributed which leads to slow eigenvalue decay!!! So options.arg tends to be large.
    options.kAuto=1;    % automatic parameter choosing
    
    preoptions.type='Exp';
    preoptions.neighbor=10;
    

    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        
        if(r>nSample)
            options.new_dim = nSample;
        else 
            options.new_dim = r;
        end
        
        try
            [Ztr{k},kpcaModel{k}] = kpcaInit(Ytr{k},options);
    %         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
            model_gp{k} = cigp_v2_03(kxtr, Ztr{k}, xte);
            Ypred{k} = Kpca_PreImage2(model_gp{k}.yTe_pred,kpcaModel{k},preoptions);
%         train_pred = Kpca_PreImage2(utrain_pred(:,1:k),kpcaModel,preoptions);
        catch 
            Ypred{k} = [];
        end
        
    end
    model.isomap = kpcaModel;
    model.gp = model_gp;
end