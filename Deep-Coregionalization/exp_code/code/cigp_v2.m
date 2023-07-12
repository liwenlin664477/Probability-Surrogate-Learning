function model = cigp_v2(xTr, yTr, xTe)
%train cigp model

    dist = pdist2(xTr,min(xTr));
    [~,index] = sort(dist);
    xTr = xTr(index,:);
    yTr = yTr(index,:);


    a0 = 1e-3; b0 = 1e-3;
    [N,d] = size(xTr);

    m = size(yTr,2);
    D = yTr*yTr';
    assert(size(xTr,1)==size(yTr,1),'inconsistent data');
    log_bta = log(1/var(yTr(:)));
%     log_bta = log(1/eps);
    
    log_l = zeros(d,1);
%     log_l = log(mean(xTr)'/10);
    
    %log_l = 2*log(median(pdist(Xtr)))*ones(d,1);
    log_sigma = 0;
    log_sigma0 = log(1e-12);
    params = [log_l;log_sigma;log_sigma0;log_bta];
    fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, 'ard'), params);
    %max_iter = 1000;
    %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
    opt = [];
    opt.MaxIter = 200;
    opt.MaxFunEvals = 10000;
    new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, 'ard'), params,opt);
    [ker_param,idx] = load_kernel_parameter(new_params, d, 'ard', 0);
    bta = exp(new_params(idx+1));
    
    model = [];
    model.params = new_params;
    model.ker_param = ker_param;
    model.bta = bta;
    
    %tr pred
    Sigma = 1/bta*eye(N) + ker_func(xTr,ker_param);
    Knn = ker_cross(xTr, xTr, ker_param);
    yTr_pred = Knn*(Sigma\yTr);
    model.yTr_pred = yTr_pred;
    
    %te pred
    if ~isempty(xTe)
        Ksn = ker_cross(xTe,xTr,ker_param);
        yTe_pred = Ksn*(Sigma\yTr);
        model.yTe_pred = yTe_pred;
    end
    
    
end

