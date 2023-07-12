%Xtr: training input, N by d matrix
%ytr: training tensor, N by m1 by m2 ... by mK tensor (output already tensorized)
%Xtest: test input
%ytest: test tensor
%rank: # of latent featurs
%a0, b0: gamma prior for bta, typically 10^-3
function model = hogp_v2(Xtr, ytr, Xtest, rank)

    a0 = 1e-3;
    b0 = 1e-3;

    nvec = size(ytr);
    nmod = length(nvec);
    [~,d] = size(Xtr);
    r = [d, rank*ones(1, nmod-1)];
%     r = ones(1, nmod)*rank;
    %latent feature initialization
    U = cell(nmod, 1);
    U{1} = Xtr;
    for k=2:nmod
        coor = linspace(-1,1,nvec(k));
        coor = (coor - mean(coor))/std(coor);
        %init with coordiates
        U{k} = repmat(coor', 1, r(k));
        %init with coordiates + small randomness
%         U{k} = U{k} + 0.0001*randn(size(U{k}));
        %random init
        %U{k} = randn(nvec(k), r(k));
%         U{k} = rand(nvec(k), r(k));
    end
    
    %init with Tucker
    %P = tucker_als_m1Fixed(ytr,r,Xtr);
    %U = P.U;

    %d = r(1);
    %init noise level
    log_bta = log(1/var(ytr(:)));
    params = [];
    for k=1:nmod
        if k>1
            params = [params;U{k}(:)];
        end
        log_l = zeros(r(k),1);
        %if k>1
        %    log_l = 2*log(median(pdist(U{k})))*ones(r(k),1);
        %end
        log_sigma0 = log(1e-4);
        %log_sigma0 = log(1);
        log_sigma = 0;
        params = [params;log_l;log_sigma;log_sigma0];
        
        %for ard-linear
        %log_alpha = 0;
        %if k>1
        %    params = [params;log_alpha];
        %end
    end
    params = [params;log_bta];
%     %gradient check, no problem
%     fastDerivativeCheck(@(params) log_evidence(params,r,a0,b0, Xtr, ytr, 'ard', 'ard'), params);
%     opt = [];
% %     opt.MaxIter = 100;
% %     opt.MaxFunEvals = 10000;
%     opt.MaxIter = 20;
%     opt.MaxFunEvals = 50;
%     new_params = minFunc(@(params) log_evidence(params, r, a0, b0, Xtr, ytr, 'ard', 'ard'), params,opt);
    
    %lbfgsb
    opt = [];
    funcl = @(params) log_evidence(params, r, a0, b0, Xtr, ytr, 'ard', 'ard');
    l = -inf * ones(numel(params), 1);
    u = inf*ones(numel(params), 1);
    u(end) = log(1000);
    opt.x0 = params;
    opt.maxIts = 100;
    [new_params, ~, ~] = lbfgsb(funcl, l, u, opt);
    
    
    
%     opt = [];
%     funcl = @(params) log_evidence(params, r, a0, b0, Xtr, ytr, 'ard', 'linear');
%     l = -inf * ones(numel(params), 1);
%     u = inf*ones(numel(params), 1);
%     u(end) = log(10);
%     opt.x0 = params;
%     opt.maxIts = 100;
%     [new_params, ~, ~] = lbfgsb(funcl, l, u, opt);
    for i = progress(1:length(Xtest))
        [yPred,model, yPred_tr] = pred_HoGP(new_params, r, Xtr, ytr, Xtest, 'ard', 'ard');  pred_var = [];
%         [yPred,model, yPred_tr] = pred_HoGP_v2(new_params, r, Xtr, ytr, Xtest, 'ard', 'ard');
%         [yPred, pred_var, model, yPred_tr] = pred_HoGP_with_var(new_params, r, Xtr, ytr, Xtest, 'ard', 'linear');

    end
    %pred_mean = pred(params, r, Xtr, ytr, Xtest);
    model.yPred = yPred;
    model.yPred_tr = yPred_tr;
    model.pred_var = pred_var;
    
%     mae = tenfun(@abs, pred_mean - ytest);
%     rmse = sqrt(mean(vec(mae.*mae)));
%     mae = mean(abs(vec(mae)));
%     mae_tr = mean(abs(vec(pred_tr - ytr)));
%     ntrain = size(Xtr,1);
%     ntest = size(Xtest,1);
%     model.mae = mae;
%     model.rmse = rmse;
%     model.mae_tr = mae_tr;
%     fprintf('ntr = %d, nte = %d, HOGP MAE = %g, RMSE = %g,  train MAE = %g\n', ntrain, ntest, mae, rmse, mae_tr);

end