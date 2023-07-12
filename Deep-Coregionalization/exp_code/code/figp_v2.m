function Model = figp_v2(xTr, yTr, xTe)
%train figp model via cigp
coreGp_func = @cigp_v2_03;

    [N,d] = size(yTr);    
    for i = 1:d
        iGpModel{i} = coreGp_func(xTr, yTr(:,i), xTe);
        yTr_pred(:,i) = iGpModel{i}.yTr_pred;
                  
        if ~isempty(xTe)
            Model.yTe_pred(:,i) = iGpModel{i}.yTe_pred;
        end
    end
    
    Model.iGpModel = iGpModel;
    Model.yTr_pred = yTr_pred;
end

