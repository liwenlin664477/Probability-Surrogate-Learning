%X: N by d
%Y: N by m1 ... by mK a tensor
function model = train_PcaFIGP_zhe(Xtr, ytr, Xtest, ytest, nBasis, tr_mean, tr_std)
    a0 = 1e-3; b0 = 1e-3;
    [N,d] = size(Xtr);
    
    %Y = tenmat(ytr,1);
    %Y = Y.data;
    Y = ytr;
    m = size(Y,2);
    assert(size(Xtr,1)==size(Y,1),'inconsistent data');
    
    if min(size(ytr))<nBasis
        nBasis = min(size(ytr));
    end
        
    
    [U,S,V] = svds(Y, nBasis);
    
    for i = 1:nBasis
        
        iu = U(:,i);
        D = iu*iu';
        
        log_bta = log(1/var(U(:,i)));
        log_l = zeros(d,1);
        log_sigma = 0;
        log_sigma0 = log(1e-2);
        params = [log_l;log_sigma;log_sigma0;log_bta];
        fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,1, Xtr, D), params);
        
        opt = [];
        opt.MaxIter = 100;
        opt.MaxFunEvals = 10000;

        new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,1, Xtr, D), params,opt);
        [ker_param,idx] = load_kernel_parameter(new_params, d, 'ard', 0);
        
%         ker_param.jitter = 1e-3;
        
        bta = exp(new_params(idx+1));
        
        model = [];
        model.ker_param = ker_param;
        model.bta = bta;
        
        %test
%         Ytest = tenmat(ytest,1);
%         Ytest = Ytest.data;
        Ksn = ker_cross(Xtest,Xtr,ker_param);
        Sigma = 1/bta*eye(N) + ker_func(Xtr,ker_param);
        uPred_mean(:,i) = Ksn*(Sigma\iu);    
        Knn = ker_cross(Xtr, Xtr, ker_param);
        utrain_pred(:,i) = Knn*(Sigma\iu);
        log_evidence(1,i) = log_evidence_CIGP(new_params,a0,b0,1, Xtr, D);
        model.uparam(:,i) = new_params;
    end
    log_evidence = sum(log_evidence);
    
    Pred_mean = uPred_mean*S*V';
    Pred_mean = Pred_mean*tr_std + tr_mean;
    
    train_pred = utrain_pred*S*V';

    %Ytest = tenmat(ytest,1);
    %Ytest = Ytest.data;
    Ytest = ytest;
    
    mae = abs(Pred_mean - Ytest);
    model.rmse = sqrt(mean(vec(mae.*mae)));
    model.mae = mean(vec(mae));
    model.mae_tr = mean(vec(abs(train_pred - Y)));
    
    model.U = U;
    model.S = S;
    model.V = V;
end
