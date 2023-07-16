function [yPred, modelRec] = train_convGp(Xtr, ytr, Xtest, rank)

%     rmpath(genpath('../code_xing'))
%     rmpath('../../code')
%     rmpath(genpath('~/Dropbox/GP-High-dimensional-output/code/'))
%     rmpath(genpath('~/Dropbox/GP-High-dimensional-output/exp/code_xing'))
    
    addpath(genpath('matlab'));
    addpath(genpath('GPmat-master'));
    addpath(genpath('netlab-master'));

%     Y = tenmat(ytr,1);
%     Y = Y.data;    

    nInduce = round(size(ytr,1) * 0.1); %20% inducing point

%     Y = reshape(ytr.data, size(ytr,1),[]);

    d = size(ytr,2);
    q = size(Xtr,2);
    
    for i = 1:d
        X{1,i} = Xtr;
    end
    ytr = num2cell(ytr,1);
     
    options = multigpOptions('fitc');
    options.kernType = 'gg';
%     options.kernType = 'rbf';
%     options.optimiser = 'scg';
    options.optimiser = 'conjgrad';
    
    options.nlf = rank;
%     options.initialInducingPositionMethod = 'randomComplete';
    options.initialInducingPositionMethod = 'randomDataIsotopic';
    
    
    options.numActive = nInduce;
    options.beta = 1e-3*ones(1, size(ytr, 2));
    options.fixInducing = false;
    
    display = 2;
    iters = 100;

    model = multigpCreate(q, d, X, ytr, options);
    % Trains the model and counts the training time
    model = multigpOptimise(model, display, iters);
    [mu, varsigma] = multigpPosteriorMeanVar(model, Xtest);
    
     
    
    
    yPred = [];
    for i = 1:d
        yPred(:,i) = mu{model.nlf+i};
        yVar(:,i) = varsigma{model.nlf+i};
    end
    

%     Ytest = reshape(ytest.data, size(ytest,1),[]);
%     
%     mae = abs(yPred - Ytest);
% 
%     modelRec.rmse = sqrt(mean(mae(:).*mae(:)));
% 
%     modelRec.mae = mean(mae(:));
    modelRec.model = model;


    modelRec.temu = mu;
    modelRec.tevar = varsigma;
    
    modelRec.yPred = yPred;
%     modelRec.yPred_tr = yPred_tr;
    modelRec.pred_var = yVar;
end
