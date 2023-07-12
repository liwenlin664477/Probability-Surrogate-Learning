function [Z,model] = mfpPCA(Y,r)
% multi fidelity progressive PCA
assert(iscell(Y),'Y must be cell');

Z=[];
nLv = length(Y);

k=1;
[Z{k}, Pca_Models{k}] = pcaInit(Y{k}, r);

for k = 2:nLv 
    nSample_k = size(Y{k}, 1);
    residual_k = Y{k} - Pca_Models{k-1}.recover(Pca_Models{k-1}.project(Y{k}));
    [Z{k}, Pca_Models{k}] = pcaInit(residual_k, r);
end

model.Pca_Models = Pca_Models;
model.recover = @(Zte) recover(Zte,Pca_Models);
model.project = @(Yte) project(Yte,Pca_Models);
end

function Zte = project(Yte,Models)
k=1;
Zte{k} = Models{k}.project(Yte{k});
for k = 2:length(Yte) 
    nSample_k = size(Yte{k}, 1);
    residual_k = Yte{k} - Yte{k-1}(1:nSample_k,:);
    Zte{k} = Models{k}.project(residual_k);
end

end

function Yte = recover(Zte,Models)
    for k = 1:length(Zte)
        Yte{k} = Models{k}.recover(Zte{k});
    end
    for k = 2:length(Zte)
        Yte{k} = Yte{k}+Yte{k-1};
    end
end