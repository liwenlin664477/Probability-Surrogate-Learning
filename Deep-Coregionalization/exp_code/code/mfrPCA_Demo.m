% mfrPCA_Demo
clear
rng(1234)
% addpath(genpath('./../../code'))
dataName = 'syn_v02';
load(dataName)

%% test with full data: no big difference
%multi level residual PCA
for r = 1:1:10
    [Z,model] = mfpPCA(Y,r);
    Y_recon = model.recover(Z);
    
    for k = 1:length(Y)
       error(r,k) =  mean((Y{k}(:)-Y_recon{k}(:)).^2);
    end
end

figure(1)
plot(log(error))

% independent pca
for r = 1:1:10
    for k = 1:length(Y)
       [z,model] = pcaInit(Y{k},r);
       Y_recon = model.recover(z);
       error2(r,k) =  mean((Y{k}(:)-Y_recon(:)).^2);
    end
end

figure(2)
plot(log(error2))

%% test with real case where ntr reduce with higher fidelity
rList =[1:10];

Ytr{1} = Y{1}(1:256,:);
Ytr{2} = Y{2}(1:128,:);
Ytr{3} = Y{3}(1:64,:);
Ytr{4} = Y{4}(1:32,:);

Yte{1} = Y{1}(257:512,:);
Yte{2} = Y{2}(257:512,:);
Yte{3} = Y{3}(257:512,:);
Yte{4} = Y{4}(257:512,:);

%multi level residual PCA
error=[];
for i = 1:length(rList)
    r = rList(i);
    [Z,model] = mfrPCA(Ytr,r);
    Zte = model.project(Yte);
    Y_recon = model.recover(Zte);
    
    for k = 1:length(Y)
       error(i,k) =  mean((Yte{k}(:)-Y_recon{k}(:)).^2);
    end
end

figure(1)
plot(log(error))

% independent pca
error2=[];
for i = 1:length(rList)
    r = rList(i);
    for k = 1:length(Y)
       [z,model] = pcaInit(Y{k},r);
       zte = model.project(Yte{k});
       Y_recon = model.recover(zte);
       error2(i,k) =  mean((Yte{k}(:)-Y_recon(:)).^2);
    end
end

figure(2)
plot(log(error2))
%% test with real case, test how many bases are need to achieve percent of energy
rList =[0.9,0.99,0.999,0.9999];

Ytr{1} = Y{1}(1:256,:);
Ytr{2} = Y{2}(1:128,:);
Ytr{3} = Y{3}(1:64,:);
Ytr{4} = Y{4}(1:32,:);

Yte{1} = Y{1}(257:512,:);
Yte{2} = Y{2}(257:512,:);
Yte{3} = Y{3}(257:512,:);
Yte{4} = Y{4}(257:512,:);

%multi level residual PCA
error=[];
for i = 1:length(rList)
    r = rList(i);
    [Z,model] = mfrPCA(Ytr,r);
    Zte = model.project(Yte);
    Y_recon = model.recover(Zte);
    
    for k = 1:length(Y)
       error(i,k) =  mean((Yte{k}(:)-Y_recon{k}(:)).^2);
       rank(i,k) = model.Pca_Models{k}.rank; 
    end
end

figure(1)
plot(log(error))

% independent pca
error2=[];
for i = 1:length(rList)
    r = rList(i);
    for k = 1:length(Y)
       [z,model] = pcaInit(Y{k},r);
       zte = model.project(Yte{k});
       Y_recon = model.recover(zte);
       error2(i,k) =  mean((Yte{k}(:)-Y_recon(:)).^2);
       rank2(i,k) = model.rank; 
    end
end

figure(2)
plot(log(error2))





