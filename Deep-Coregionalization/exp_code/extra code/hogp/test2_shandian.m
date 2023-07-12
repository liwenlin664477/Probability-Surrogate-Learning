close all;clc;
rng(2);
addpath_recurse('./tensor_toolbox_2.6');
addpath_recurse('./minFunc_2012');
addpath_recurse('./L-BFGS-B-C');
addpath_recurse('./lightspeed');
addpath_recurse('./util');


nTrainList = linspace(1,7,7);
nTrainList = 2.^nTrainList;
ntest = 100;
level = 0;

% Xall = rand(3,70);
% Yall = rand(100,100,100,70);
% 
% load('canti_10x10.mat');
% Xall = x;
% Yall = y;
% addpath('../../../../dataRepo')
load('../../data/ns_v2_01_P.mat');
Xall = X';
Yall = Y;

nTrainList = [32];

%%
for i=1:numel(nTrainList)
    ntrain = nTrainList(i);
    [Xtr, ytr, Xtest, ytest] = dateLoad(Xall,Yall,ntrain,ntest,level);
    
    % FIGP = train_FIGP(Xtr, ytr, Xtest, ytest, 1e-3, 1e-3);
    CIGP = train_CIGP(Xtr,ytr,Xtest,ytest);
    HOGP = train_HOGP(Xtr, ytr, Xtest, ytest, 1, 1e-3, 1e-3);
    %HOGP_pr = train_HOGP_pr(Xtr, ytr, Xtest, ytest, 5, 1, 1e-3, 1e-3);
  
    
   
end



%% 
function [Xtr, ytr, Xtest, ytest] = dateLoad(Xall,Yall,ntrain,ntest,level)
n_total = size(Xall,2);
% ntrain = 128;
% ntest = 32;
ind = randperm(n_total);
Xtr = Xall(:,ind(1:ntrain));
Xtr = Xtr';
ytr = Yall(:,:,2:2:100,ind(1:ntrain));
Xtest = Xall(:, ind(ntrain+1:ntrain+ntest));
Xtest = Xtest';
ytest = Yall(:,:,2:2:100,ind(ntrain+1:ntrain+ntest));
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
% level = 0.5;%0.1;
ytr = ytr + (level)*randn(size(ytr));
%ytest = ytest + (level)*randn(size(ytest));
%reshape data to tensor, rotate mode, input X, is in the first mode
ytr = permute(tensor(ytr),[4,1,2,3]);
ytest = permute(tensor(ytest), [4,1,2,3]);

% ytr = permute(tensor(ytr),[3,1,2]);
% ytest = permute(tensor(ytest), [3,1,2]);


end