%plus plosterior regularizer on test data as well
%param: a vector of parameters, inculding kernel paramters, noise inverse variance 
%a0, b0: gamma prior for bta, e.g., 10^-3
%X: training input, N by d matrix
%Xtest: test input, Ntest by d matrix
%Y: training output, N by m1 by m2 ... by mK tensor (output already tensorized)
%r: rank setting, r(1) = d, r(2:K) is the rank of latent features
%lam: regularization strength
%output: logL: log model evidence, dLogL: gradient
function [f, df] = log_evidence_pr(params, r, a0, b0, X, Y, Xtest, lam, ker_type1, ker_type2)
    [N,d] = size(X);
    [Nt,~] = size(Xtest);
    nvec = size(Y); %it must be [N, m1, m2, ..., mK]
    assert(N==nvec(1), 'inconsistent input-output');
    nmod = length(nvec);
    %each dimension has a diffrent kernel, let us assume ARD first
    ker_params = cell(nmod,1);
    U = cell(nmod, 1);
    U{1} = X;
    %extract parameters
    [ker_params{1},idx] = load_kernel_parameter(params, d, ker_type1, 0);
    for k=2:nmod
        U{k} = reshape(params(idx+1:idx+nvec(k)*r(k)),nvec(k), r(k));
        [ker_params{k},idx] = load_kernel_parameter(params, r(k), ker_type2, idx+nvec(k)*r(k));
    end
    bta = exp(params(idx+1));
    Sigma = cell(nmod, 1);Lam = cell(nmod, 1);LamDiag = cell(nmod,1);V = cell(nmod,1);Vt = cell(nmod,1);
    for k=1:nmod
        Sigma{k} = ker_func(U{k}, ker_params{k});
        [V{k}, LamDiag{k}] = eig(Sigma{k});
        Lam{k} = diag(LamDiag{k});
        Vt{k} = V{k}';
    end
    logL = (a0 - 1)*log(bta) - b0*bta;
    dbta = (a0 - 1)/bta - b0;
    %log|bta^{-1} + \Sigma \kron ...|
    btaInvPlusSigma = 1/bta + tensor(ktensor(Lam));
    M = 1./btaInvPlusSigma;
    M12 = tenfun(@sqrt,M);
    logL = logL - 0.5*sum(log(btaInvPlusSigma(:)));
    %sum(btaInvPlusSigma(:)<0)
    dbta = dbta + 0.5*bta^(-2)*sum(M(:));
    dker_params = cell(nmod,1);
    dU = cell(nmod, 1);
    for k=1:nmod
        bk = ttv(M,Lam,setdiff(1:nmod,k));
        Ak = -0.5*V{k}*diag(bk.data)*V{k}';
        [dU{k}, dker_params{k}] =  ker_grad(U{k}, Ak, Sigma{k},ker_params{k});        
    end
    %-0.5vec(Y)^\top (bta^-1 I + \Sigma)^{-1} vec(Y)
    T = times(M12, ttm(Y, Vt));
    T2 = times(M12,T);
    D12Y = ttm(T, V);
    DY = ttm(T2, V);
    logL = logL - 0.5*sum(D12Y(:).*D12Y(:));
    dbta = dbta - 0.5*bta^(-2)*sum(DY(:).*DY(:));    
    %C is T2;
    for k=1:nmod
        Fk = ttm(T2,LamDiag,setdiff(1:nmod, k));
        Fks = tenmat(Fk, k);
        Cs = tenmat(T2, k);
        Ak = 0.5*(V{k}*(Cs.data*Fks.data')*Vt{k});
        [dUk, dkp] = ker_grad(U{k}, Ak, Sigma{k}, ker_params{k});
        dker_params{k} = dker_params{k} + dkp;
        if k>1
            dU{k} = dU{k} + dUk;    
        end
    end
    %%%%Predictive Variance (post. regularization part)
    dU_R = dU;
    dker_params_R = dker_params;
    for k=1:nmod
        dU_R{k} = zeros(size(dU_R{k}));
        dker_params_R{k} = zeros(size(dker_params_R{k}));
    end
    %K** = Ktt \kron Sigma^2 ... \kron Sigma^K
    Ktt = ker_func(Xtest, ker_params{1});
    %trace(A \kron B) = trace(A)trace(B)
    R = trace(Ktt);
    for k=2:nmod
        R = R*trace(Sigma{k});
    end
    [~, dkp] = ker_grad(Xtest, R/trace(Ktt)*eye(Nt), Ktt, ker_params{1});
    dker_params_R{1} = dker_params_R{1} + dkp;
    for k=2:nmod
        [dUk, dkp] = ker_grad(U{k}, R/trace(Sigma{k})*eye(nvec(k)), Sigma{k}, ker_params{k});
        dker_params_R{k} = dker_params_R{k} + dkp;
        dU_R{k} = dU_R{k} + dUk;
    end
    %-K*n(Knn+bta^{-1}I)^{-1}Kn*
    Ktn = ker_cross(Xtest, X, ker_params{1});
    V11 = V{1}*diag(1./Lam{1});
    E12 = (Ktn*V11)';
    diagE = sum(E12.^2, 2);
    onesCell = cell(nmod, 1);
    Lam2 = cell(nmod, 1);
    for k=1:nmod
        onesCell{k} = ones(nvec(k),1);
        Lam2{k} = Lam{k}.^2;
    end
    onesCell{1} = diagE;
    Lam2Kt = tensor(ktensor(Lam2));
    D = M.*Lam2Kt;
    term2 = ttv(D, onesCell);
    R = R - term2;
    %grad. part
    %dKtn
    u = ttv(D, onesCell, 2:nmod);
    A = (V11*diag(u.data)*V11')*Ktn';
    [~, dkp] = ker_cross_grad(-2*A, Ktn, Xtest, X, ker_params{1});
    dker_params_R{1} = dker_params_R{1} + dkp;
    %dSigma^{(k)}
    Lam1var = Lam;
    Knttn = Ktn'*Ktn;
    Lam1var{1} = diag(V11'*Knttn*V{1});
    LamKt = tensor(ktensor(Lam));
    F = M.*LamKt;
    for k=2:nmod
        dnk = ttv(F, Lam1var, setdiff(1:nmod,k));
        A = V{k}*diag(dnk.data)*Vt{k};
        [dUk, dkp] = ker_grad(U{k}, -2*A, Sigma{k}, ker_params{k});
        dU_R{k} = dU_R{k} + dUk;
        dker_params_R{k} = dker_params_R{k} + dkp;
    end
    %dbta    
    term = ttv(D.*M, onesCell);
    dbta_R = - 1/(bta^2)*term;    
    %dSigma{k} in -trace(A dS A') -- see write-up
    %this is only an approximation!
    Lam2{1} = diag(Vt{1}*Knttn*V{1});
    Lam2Kt = tensor(ktensor(Lam2));
    T = (M.^2).*Lam2Kt;
    for k=1:nmod
        dnk = ttv(T, Lam, setdiff(1:nmod, k));
        A = -V{k}*diag(dnk.data)*Vt{k};
        [dUk, dkp] =ker_grad(U{k}, -A, Sigma{k}, ker_params{k});
        dker_params_R{k} = dker_params_R{k} + dkp;
        if k>1
            dU_R{k} = dU_R{k} + dUk;
        end
    end
    
    %add the regularization part
    lam = Nt/N*lam;
    logL = logL - lam*R;
    dbta = dbta - lam*dbta_R;
    for k=1:nmod
        dker_params{k} = dker_params{k} - lam*dker_params_R{k};
        if k>1
            dU{k} = dU{k} - lam*dU_R{k};
        end
    end
    %assemble gradients
    d_log_bta = bta*dbta;
    
    %regularizaton
    logL = logL - 0.5*sum(params(1:end-1).*params(1:end-1));% - 0.5*bta*bta;
    %d_log_bta = d_log_bta - bta*bta;
    f = -logL;
    df = zeros(numel(params),1);
    idx = 0;
    df(idx+1:idx+length(dker_params{1})) = dker_params{1};
    idx = idx + length(dker_params{1});
    for k=2:nmod
        df(idx+1:idx+numel(dU{k})) = dU{k}(:);
        idx = idx + numel(dU{k});
        df(idx+1:idx+numel(dker_params{k})) = dker_params{k};
        idx = idx + numel(dker_params{k});
    end
    %if bta>10
    %    d_log_bta = 0;
    %end
    df(idx+1) = d_log_bta;  
    %regularizaton
    df(1:end-1) = df(1:end-1) - params(1:end-1);
    df = -df;
end