function err = err_eval(yPred,yTrue)

    if isempty(yPred)
        yPred = nan(size(yTrue));
    end
    
    %% mse
    err2 = (yPred - yTrue).^2;
%     err.mse_sample = mean(err2,2);
%     err.mse_dims = mean(err2, 1);     
             
%     err2 = mean((yPred - yTrue).^2, 2);   
%     err.mse_sample = err2;
%     err.mse_dims = mean((yPred - yTrue).^2, 1); 
    
    err.mse = mean2(err2);
    err.mse_std = std2(err2);
    
    err.mse_dims = mean(err2);
    err.mse_std_dims = std(err2);
    
    err.rmse = sqrt(err.mse);
    err.rmse_std = std2(sqrt(err2));
    
    %% mae
    err_abs = abs(yPred - yTrue);
    
    err.mae = mean2(err_abs);
    err.mae_std = std2(err_abs);
    
    err.mae_dims = mean(err2);
    err.mae_std_dims = std(err2);
    
    %% relative err
    re = sqrt( mean(err2,2) ./ (mean(yTrue.^2, 2)+eps));
    err.mre = mean(re);
    err.mre_std = std(re);
    
    yMean = mean(yTrue(:));
    yStd = std(yTrue(:));
    
    %% normalize error
    yPred = (yPred - yMean)/yStd;
    yTrue = (yTrue - yMean)/yStd;
    
    nerr2 = (yPred - yTrue).^2;
    
    err.nmse = mean2(nerr2);
    err.nmse_std = std2(nerr2);
    
    err.nmse_dims = mean(nerr2);
    err.nmse_std_dims = std(nerr2);

    err.nrmse = sqrt(mean2(nerr2));
    err.nrmse_std = sqrt(std2(nerr2));
end