%Xtr: training input, N by d matrix
%ytr: training tensor, N by m1 by m2 ... by mK tensor (output already tensorized)
%Xtest: test input
%ytest: test tensor
%rank: # of latent featurs
%a0, b0: gamma prior for bta, typically 10^-3
function model = train_HOGP_pr(Xtr, ytr, Xtest, ytest, rank, lam, a0, b0)
    nvec = size(ytr);
    nmod = length(nvec);
    [~,d] = size(Xtr);
    r = [d, rank*ones(1, nmod-1)];
    %latent feature initialization
    U = cell(nmod, 1);
    U{1} = Xtr;
    for k=2:nmod
        coor = linspace(-1,1,nvec(k));
        coor = (coor - mean(coor))/std(coor);
        %init with coordiates
        U{k} = repmat(coor', 1, r(k));
        %init with coordiates + small randomness
        U{k} = U{k} + 0.001*randn(size(U{k}));
        %random init
        %U{k} = randn(nvec(k), r(k));
        U{k} = rand(nvec(k), r(k));
    end
    
    %init with Tucker
    %P = tucker_als_m1Fixed(ytr,r,Xtr);
    %U = P.U;

    log_bta = log(1/var(ytr(:)));
    params = [];
    for k=1:nmod
        if k>1
            params = [params;U{k}(:)];
        end
        log_l = zeros(r(k),1);
        %log_l = 2*log(median(pdist(U{k})))*ones(d,1);
        log_sigma = 0;
        log_sigma0 = log(1e-4);
        %log_sigma0 = log(1);
        %if k>1
        %    log_l = log(0.05)*ones(r(k),1);
        %end
        params = [params;log_l;log_sigma;log_sigma0];
    end
    params = [params;log_bta];
    %gradient check, no problem
    fastDerivativeCheck(@(params) log_evidence_pr(params,r,a0,b0, Xtr, ytr, Xtest, lam, 'ard', 'ard'), params);
    %rst = gradientcheck(@(params) log_evidence(params,r,1e-3,1e-3, Xtr, ytr), params);
    %max_iter = 1000;
    %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
    opt = [];
    opt.MaxIter = 100;
    opt.MaxFunEvals = 10000;
    %opt.Method = 'scg';
    new_params = minFunc(@(params) log_evidence_pr(params, r, a0, b0, Xtr, ytr, Xtest, lam, 'ard', 'ard'), params,opt);
    [pred_mean,model, pred_tr] = pred_HoGP(new_params, r, Xtr, ytr, Xtest, 'ard', 'ard');
    %pred_mean = pred(params, r, Xtr, ytr, Xtest);
    mae = tenfun(@abs, pred_mean - ytest);
    rmse = sqrt(mean(vec(mae.*mae)));
    mae = mean(abs(vec(mae)));
    mae_tr = mean(abs(vec(pred_tr - ytr)));
    ntrain = size(Xtr,1);
    ntest = size(Xtest,1);
    model.mae = mae;
    model.rmse = rmse;
    model.mae_tr = mae_tr;
    fprintf('ntr = %d, nte = %d, HOGP MAE = %g, RMSE = %g,  train MAE = %g\n', ntrain, ntest, mae, rmse, mae_tr);

end