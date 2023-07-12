function [Ypred, model] = IsomapGp_v01(xtr,Ytr,xte,r)
% use normal y(:)

    options.dim_new=r;                % New dimension
    options.neighborType='k';               % Type of neighbor.Choice:1)'k';Choice:2)'epsilon'
    options.neighborPara=10;                % parameter for choosing the neighbor. number of neighbor for "k" type and radius for 'epsilon'
    options.metric='euclidean';             % Method of measurement. Metric    
    
    preoptions.ReCoverNeighborType='k';     % Type of neighbor of new point. Choice:1)'k';Choice:2)'epsilon'
%     preoptions.ReCoverNeighborPara=10;      % Parameter of neighbor of new point
    

    nSample = size(Ytr, 1);
    
        if(r > nSample)
            options.dim_new = nSample;
        else 
            options.dim_new = r;
        end
        
        
        try
            [Ztr,isomapModel] = Isomaps(Ytr,options);
%             [Ztr{k},isomapModel{k}] = isoma(Ytr{k},options);
    %         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
            model_gp = cigp_v2_03(xtr, Ztr, xte);
            Ypred = Isomaps_PreImage(model_gp.yTe_pred,isomapModel,preoptions);
%         train_pred = Isomaps_PreImage(utrain_pred(:,1:k),kpcaModel,preoptions);
        catch
            Ypred = [];
        end
        
    model.isomap = isomapModel;
    model.gp = model_gp;
end