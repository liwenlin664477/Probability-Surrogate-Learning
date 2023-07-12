close all;clc;
rng(0);
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./minFunc_2012');
addpath_recurse('./lightspeed');
addpath_recurse('./util');

load('canti_10x10.mat');
Xall = x;
Yall = y;
n_total = size(x,2);
ntrain = 256;
ntest = 32;
ind = randperm(n_total);
Xtr = Xall(:,1:ntrain);
Xtr = Xtr';
ytr = Yall(:,:,1:ntrain);
Xtest = Xall(:, ntrain+1:ntrain+ntest);
Xtest = Xtest';
ytest = Yall(:,:,ntrain+1:ntrain+ntest);
%normalization
[Xtr, Xtest] = normalize_standard(Xtr, Xtest);
ytr_vec = ytr(:);
ytest_vec = ytest(:);
ytr_mean = mean(ytr_vec);ytr_std = std(ytr_vec);
ytr_vec = (ytr_vec - ytr_mean)/ytr_std;
ytest_vec = (ytest_vec - ytr_mean)/ytr_std;
ytr = reshape(ytr_vec, size(ytr));
ytest = reshape(ytest_vec, size(ytest));
%add extra noise
level = 0;%0.1;
ytr = ytr + (level)*randn(size(ytr));
%reshape data to tensor, rotate mode, input X, is in the first mode
ytr = permute(tensor(ytr),[3,1,2]);
ytest = permute(tensor(ytest), [3,1,2]);
%test CIGP
%FIGP = train_FIGP(Xtr, ytr, Xtest, ytest, 1e-3, 1e-3);
CIGP = train_CIGP(Xtr,ytr,Xtest,ytest);
HOGP = train_HOGP(Xtr, ytr, Xtest, ytest, 2, 1e-3, 1e-3);
%HOGP = train_HOGP_pr(Xtr, ytr, Xtest, ytest, 2, 0, 1e-3, 1e-3);

%initialization
% coor = linspace(-1,1,10);
% coor = (coor - mean(coor))/std(coor);
% r = [3,2,2];
% nmod = 3;
% %latent feature initialization
% U = cell(nmod, 1);
% U{1} = Xtr;
% U{2} = repmat(coor',1,r(2));
% U{3} = repmat(coor',1,r(3));
% U{2} = U{2} + 0.001*randn(size(U{2}));
% U{3} = U{3} + 0.001*randn(size(U{3}));
% %random init
% U{2} = randn(10,r(2));
% U{3} = randn(10, r(3));
% %init with Tucker
% %P = tucker_als_m1Fixed(ytr,r,Xtr);
% %U = P.U;
% 
% d = r(1);
% %init noise level
% log_bta = log(1/var(ytr(:)));
% params = [];
% for k=1:nmod
%     if k>1
%         params = [params;U{k}(:)];
%     end
%     log_l = zeros(r(k),1);
%     %log_l = 2*log(median(pdist(U{k})))*ones(d,1);
%     log_sigma = 0;
%     log_sigma0 = log(1e-4);
%     %if k>1
%     %    log_l = log(0.05)*ones(r(k),1);
%     %end
%     params = [params;log_l;log_sigma;log_sigma0];
% end
% params = [params;log_bta];
% %gradient check, no problem
% fastDerivativeCheck(@(params) log_evidence(params,r,1e-3,1e-3, Xtr, ytr), params);
% %rst = gradientcheck(@(params) log_evidence(params,r,1e-3,1e-3, Xtr, ytr), params);
% %max_iter = 1000;
% %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
% opt = [];
% opt.MaxIter = 500;
% opt.MaxFunEvals = 10000;
% new_params = minFunc(@(params) log_evidence(params, r, 1e-3, 1e-3, Xtr, ytr), params,opt);
% 
% [pred_mean,model] = pred_HoGP(new_params, r, Xtr, ytr, Xtest);
% %pred_mean = pred(params, r, Xtr, ytr, Xtest);
% mae = tenfun(@abs, pred_mean - ytest);
% mae = mean(abs(mae(:)));
% fprintf('ntr = %d, nte = %d, HOGP MAE = %g\n', ntrain, ntest, mae);


