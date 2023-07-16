% intTest1
clear

nTotal = 300;
randId = randperm(nTotal);

trainId = randId(1:50);
testId = randId(251:300);

q = 4;
d = 2;

xTemp = rand(300,q);
yTemp(:,1) = sin(sum(xTemp,2));
% yTemp(:,2) = sin(yTemp(:,1));
yTemp(:,2) = sum(xTemp,2).^2;

% xTemp = linspace(0,3,nTotal)';
% yTemp(:,1) = sin(xTemp);
% yTemp(:,2) = (xTemp.^2);


% X = num2cell(xtemp(trainId,:),1)';
% Y = num2cell(ytemp(trainId,:),1)';
% 
% Xt = num2cell(xtemp(testId,:),1)';
% Yt = num2cell(ytemp(testId,:),1)';
%     for i = 1:q
%         X{i,1} = xtemp(trainId,i);
%         Xt{i,1} = xtemp(testId,i);
%     end
% 
% 
% for i = 1:d
%     X{i,1} = xtemp(trainId,i);
%     Y{i,1} = ytemp(trainId,i);
% 
%     Xt{i,1} = xtemp(testId,i);
%     Yt{i,1} = ytemp(testId,i); 
% end

for i = 1:d
    X{1,i} = xTemp(trainId,:);
    Y{1,i} = yTemp(trainId,i);

    Xt{1,i} = xTemp(testId,:);
    Yt{1,i} = yTemp(testId,i); 
end

% X = xtemp(trainId,:);
% Y = ytemp(trainId,:);
% Xt = xtemp(testId,:);
% Yt = ytemp(testId,:);


% Set up model
% options = multigpOptions('ftc');
% options.optimiser = 'conjgrad';
% % options.kernType = 'lfm';
% options.kernType = 'gg';
% options.tieOptions.selectMethod = 'free';

options = multigpOptions('fitc');
options.kernType = 'gg';
options.optimiser = 'scg';
options.nlf = 2;
options.initialInducingPositionMethod = 'randomComplete';
options.numActive = 20;
options.beta = 1e-3*ones(1, size(yTemp, 2));
options.fixInducing = false;


%%
% for i = 1:options.nlf
%     X{1,i} = 0;
%     Y{1,i} = 0;
% end
% for i = 1:size(yTemp, 2)
%     Y{1,i+options.nlf} = yTemp(:, i);
%     X{i+options.nlf} = xTemp;            
% end


%%

% q = 1;
% d = size(ytemp, 2);

% Creates the model
model = multigpCreate(q, d, X, Y, options);
%model.scale = repmat(sqrt(sum(var(y))/model.d), 1, model.d);

% model.scale = repmat(1, 1, model.d);
% model.m = multigpComputeM(model);


display = 2;
iters = 1000;

% Trains the model and counts the training time
model = multigpOptimise(model, display, iters);

% Xt = linspace(min(X{model.nlf+1})-0.5,max(X{model.nlf+1})+0.5,200)';


[mu, varsigma] = multigpPosteriorMeanVar(model, Xt);

yPred=cell(d,1);
for i = 1:d
    yPred{i,:} = mu{model.nlf+i};
end

%% 
figure(1)
scatter(yPred{1},Yt{1})
refline(1)

figure(2)
scatter(yPred{2},Yt{2})
refline(1)

% figure(1)
% scatter(xTemp(testId,1), yPred{1})
% hold on 
% plot(xTemp(:,1),yTemp(:,1))
% hold off
% 
% figure(2)
% scatter(xTemp(testId,1), yPred{2})
% hold on 
% plot(xTemp(:,1),yTemp(:,2))
% hold off
