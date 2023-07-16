% TEST_v02
% v02: run multiple exp setting
clear
rng(1234)
% addpath(genpath('./../../code'))
dataName = 'syn_v02';
load(dataName)

%% main 
trRatio = 2;
nLv = 4;
nte = 128;
trRatio = 2;
seed = 1;

ntr_lvTop_list = 2.^[3:1:5];
r_list = [2:2:10];


for i = 1:length(ntr_lvTop_list)
    ntr_lvTop = ntr_lvTop_list(i);
    [xtr,Ytr,xte,yte_golden] = ttLvGen(X, Y, nLv, ntr_lvTop, nte, trRatio, seed);
    
    for j = 1:length(r_list)
        r = r_list(j);
        
        error{i,j} = test_all(xtr,Ytr,xte,yte_golden,r);

    end
    
end
    
errPlot(error)


%%
function [Ypred, model] = deepPcaGp_v01(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp)

    [Ztr,U] = multiLvPca(Ytr, r, iMethod_pca);
%     [Ztr,U] = multiLvPca_v2(Ytr, r, iMethod_pca);
%     [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
    [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
    [Zpred, model] = deepGp_v01(xtr,Ztr,xte, iMethod_dgp);
    
    Ypred = multiLvPcaInv(Zpred, U, iMethod_pca);
%     for k = 1:length(Ytr)
%        Ypred{k} = Zpred{k} * U{k};
%     end
    
    model.U = U;
    model.Ztr = Ztr;
end


function [Ypred, model] = pcaLvGp(xtr,Ytr,xte,r)

    for k = 1:length(Ytr)
        [U{k}, Ztr{k}] = myPca(Ytr{k}, r);
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        model{k} = cigp_v2(kxtr, Ztr{k}, xte);
    
        Ypred{k} = model{k}.yTe_pred * U{k};
    end

end

function error = err(yPred,yTrue)

    err2 = mean((yPred - yTrue).^2, 2);
    error.mse = mean(err2);
    error.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    error.re = mean(re);
    error.re_std = std(re);

end

function [xtr,ytr,xte,yte] = ttGen(y, x, ntr, nte, seed)
% training & testing data generate
    assert(size(x,1) == size(y,1),'N samble not match');
    N = size(x,1);
    assert((ntr + nte)<=N,'ntr + nte > N sample');
    
    rng(seed);
    id_rand = randperm(N);
    
    id_tr = id_rand(1:ntr);
    id_te = id_rand(N-nte+1:N);

    xtr = x(id_tr, :);
    xte = x(id_te, :);
    
    ytr = y(id_tr, :);
    yte = y(id_te, :);
end

function [xtr,Ytr,xte,yte_golden] = ttLvGen(x, Y, nLv, ntr_top, nte, trRatio, seed)
% training & testing data generate
    rng(seed)
    N = size(x,1);
    
    nTr{nLv} = ntr_top;
    for i = nLv:-1:2
        nTr{i-1} = nTr{i}*trRatio;
    end
    assert((nTr{1} + nte)<=N,'nTr{1} + nte > N sample');
    
    %shuffle data
    id_rand = randperm(N);
    id_tr = id_rand(1:(N-nte));
    id_te = id_rand(N-nte+1:N);

    xtr = x(id_tr, :);
    xte = x(id_te, :);

    for k = 1:nLv
        Ytr{k} = Y{k}(id_tr(1:nTr{k}),:);
    end
    yte_golden = Y{nLv}(id_te,:);
    
end


function error = exp_v1(x,Y,nlv,nte,ntr,r,trRatio,seed)
% nlv number of lv to be included. minimum 2.
    rng(seed)
    
    N = size(x,1);
    
    nTr{nlv} = ntr;
    for i = nlv:-1:2
        nTr{i-1} = nTr{i}*trRatio;
    end
    
    %shuffle data
    id_rand = randperm(N);
    id_tr = id_rand(1:ntr);
    id_te = id_rand(N-nte+1:N);

    xtr = x(id_tr, :);
    xte = x(id_te, :);

    for k = 1:nLv
        Ytr{k} = Y{k}(id_tr(1:nTr{k}),:);
    end
    yte_golden = Y{nLv+1}(id_te,:);
    
    error = test_all(xtr,Ytr,xte,yte_golden,r);

end

function error = test_all(xtr,Ytr,xte,yte_golden,r)

    k=1;
    for iMethod_pca = 2:4
        for iMethod_dgp = 1:3     
            try
%                 [Ypred, model] = deepPcaGp_v01(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
                [Ypred, model] = deepPcaGp_v03(xtr,Ytr,xte,r, iMethod_pca, iMethod_dgp);
                error(k) = err(Ypred{end},yte_golden);  
            end
            k=k+1;
        end
    end
    %%
    [Ypred, model] = pcaLvGp(xtr,Ytr,xte,r);
    for i = 1:length(Ytr)
        error(k) = err(Ypred{i},yte_golden);  
        k=k+1;
    end

end

function errPlot(error)
% i : collapse direction
    for i = 1:size(error,1)
        for j = 1:size(error,2)
            try
                errM(i,j,:) = [error{i,j}.re];
                stdM(i,j,:) = [error{i,j}.re_std];
            end
        end    
    end
    markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h','+b'};
    
    
    close all
    for iMethod = 1:size(errM,3)
        ierrM = squeeze(errM(:,:,iMethod));
        istdM = squeeze(stdM(:,:,iMethod));
        
        for i = 1:size(ierrM,1)
            figure(i)
            hold on
            errorbar(ierrM(i,:),istdM(i,:),markers{iMethod})  
        end
            

        for j = 1:size(ierrM,2)
            figure(j+100)
            hold on
            errorbar(ierrM(:,j),istdM(:,j),markers{iMethod})  
        end
                
        
    end
    
end

function errPlot_2(error)
% i : collapse direction
    for i = 1:size(error,1)
        for j = 1:size(error,2)
            try
                errM(i,j,:) = [error{i,j}.re];
                stdM(i,j,:) = [error{i,j}.re_std];
            end
        end    
    end
    markers = {'+-','o-','*-','.-','x-','s-','d-','^-','v-','>-','<-','p-','h-','+b-'};
    
    
    close all
    for iMethod = 1:size(errM,3)
        ierrM = squeeze(errM(:,:,iMethod));
        istdM = squeeze(stdM(:,:,iMethod));
        
        for i = 1:size(ierrM,1)
            figure(i)
            hold on
            plot(ierrM(i,:),markers{iMethod})  
        end
            

        for j = 1:size(ierrM,2)
            figure(j+100)
            hold on
            plot(ierrM(:,j),markers{iMethod})  
        end
                
        
    end
    
    
    
end


