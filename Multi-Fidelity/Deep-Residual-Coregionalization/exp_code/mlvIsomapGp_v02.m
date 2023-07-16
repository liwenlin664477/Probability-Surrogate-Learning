function [Ypred, model] = mlvIsomapGp_v02(xtr,Ytr,xte,r)

    options.dim_new=r;                % New dimension
    options.neighborType='k';               % Type of neighbor.Choice:1)'k';Choice:2)'epsilon'
    options.neighborPara=10;                % parameter for choosing the neighbor. number of neighbor for "k" type and radius for 'epsilon'
    options.metric='euclidean';             % Method of measurement. Metric    
    
    preoptions.ReCoverNeighborType='k';     % Type of neighbor of new point. Choice:1)'k';Choice:2)'epsilon'
%     preoptions.ReCoverNeighborPara=10;      % Parameter of neighbor of new point
    

    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        
        if(r > nSample)
            options.dim_new = nSample;
        else 
            options.dim_new = r;
        end
        
        
        
         
        [Ztr{k},isomapModel{k}] = Isomaps(Ytr{k},options);
        
%         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
%         model_gp{k} = cigp_v2_02(kxtr, Ztr{k}, xte);
        model_gp{k} = figp_v2(kxtr, Ztr{k}, xte);
        
        Ypred{k} = Isomaps_PreImage(model_gp{k}.yTe_pred,isomapModel{k},preoptions);
%         train_pred = Isomaps_PreImage(utrain_pred(:,1:k),kpcaModel,preoptions);
        
    end
    model.isomap = isomapModel;
    model.gp = model_gp;
end