




%% function model = ivmOptimiseIvm(model, display)
dVal = model.d;
for k = 1:dVal  %dVal is the number of latent
    
  [indexSelect, infoChange(k)] = ivmSelectPoint(model);
  dataIndexSelect = model.J(indexSelect);
  model = ivmAddPoint(model, dataIndexSelect);

  
end


%% Optimization ivm wrt hyperparameter as a whole to make clear 

model = optimiseParams('kern', 'scg', 'ivmKernelObjective', ...
                       'ivmKernelGradient', options, model);

function f = ivmKernelObjective(params, model)
%/~
if any(isnan(params))
  warning('Parameter is NaN')
end
%~/

model.kern = kernExpandParam(model.kern, params);
f = ivmApproxLogLikelihood(model);
f = f + kernPriorLogProb(model.kern);
f = -f;

end

function L = ivmApproxLogLikelihood(model);

x = model.X(model.I, :);
m = model.m(model.I, :);
K = kernCompute(model.kern, x);
L = 0;

if model.noise.spherical
  % there is only one value for all beta
  [invK, UC] = pdinv(K+diag(1./model.beta(model.I, 1)));
  logDetTerm = logdet(K, UC);
end
  
for i = 1:size(m, 2)
  if ~model.noise.spherical
    [invK, UC] = pdinv(K+diag(1./model.beta(model.I, i)));
    logDetTerm = logdet(K, UC);
  end
  L = L -.5*logDetTerm- .5*m(:, i)'*invK*m(:, i);
end
end
