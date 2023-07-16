function [yPred, model] = mlvPcaGp_v03(xtr,Ytr,xte,r)
% gp_func = @cigp_v2_03 normalize y(:) and x

    for k = 1:length(Ytr)        

            nSample = size(Ytr{k}, 1);
            kxtr = xtr(1:nSample,:);  

            if(r>nSample)
                r = nSample;
            end
            
            try 
                [yPred{k},model{k}] = pcaGp(kxtr,Ytr{k},xte,r);
            catch 
                yPred{k} = [];
            end

    end

end

function [yPred,model] = pcaGp(xtr,ytr,xte,r)

gp_func = @cigp_v2_03;

[ztr,model_pca] = pcaInit(ytr,r);

model_gp = gp_func(xtr, ztr, xte);
yPred = model_pca.recover(model_gp.yTe_pred);


model.model_gp = model_gp;
model.model_pca = model_pca;
end