%X: N by d
%Y: N by m1 ... by mK a tensor
function model = train_CIGP(Xtr, ytr, Xtest, ytest)
    a0 = 1e-3; b0 = 1e-3;
    [N,d] = size(Xtr);
    Y = tenmat(ytr,1);
    Y = Y.data;
    m = size(Y,2);
    D = Y*Y';
    assert(size(Xtr,1)==size(Y,1),'inconsistent data');
    log_bta = log(1/var(ytr(:)));
    log_l = zeros(d,1);
    %log_l = 2*log(median(pdist(Xtr)))*ones(d,1);
    log_sigma = 0;
    log_sigma0 = log(1e-4);
    params = [log_l;log_sigma;log_sigma0;log_bta];
    fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, Xtr, D), params);
    %max_iter = 1000;
    %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
    opt = [];
    opt.MaxIter = 100;
    opt.MaxFunEvals = 10000;
    new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, Xtr, D), params,opt);
    [ker_param,idx] = load_kernel_parameter(new_params, d, 'ard', 0);
    bta = exp(new_params(idx+1));
    model = [];
    model.ker_param = ker_param;
    model.bta = bta;
    %test
    Ytest = tenmat(ytest,1);
    Ytest = Ytest.data;
    Ksn = ker_cross(Xtest,Xtr,ker_param);
    Sigma = 1/bta*eye(N) + ker_func(Xtr,ker_param);
    pred_mean = Ksn*(Sigma\Y);
    mae = abs(pred_mean - Ytest);
    rmse = sqrt(mean(vec(mae.*mae)));
    mae = mean(vec(mae));
    Knn = ker_cross(Xtr, Xtr, ker_param);
    train_pred = Knn*(Sigma\Y);
    train_mae = mean(vec(abs(train_pred - Y)));
    model.mae = mae;
    model.rmse = rmse;
    model.mae_tr = train_mae;
    fprintf('ntr = %d, nte = %d, CIGP MAE = %g, RMSE = %g,  training MAE = %g\n', N, size(Xtest,1), mae, rmse, train_mae);
end