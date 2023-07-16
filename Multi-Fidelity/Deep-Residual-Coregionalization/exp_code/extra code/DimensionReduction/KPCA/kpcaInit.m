function [z,model] = kpcaInit(X,options)
% KPCA Kernel Principal Component Analysis.
% a new architecture
%  
% Synopsis:
%  [Z,model] = Kpca(X)
%  [Z,model] = kpca(X,options)
%
% Description:
%  This function is implementation of Kernel Principal Component 
%  Analysis (KPCA) [Schol98b]. The input data X are non-linearly
%  mapped to a new high dimensional space induced by prescribed
%  kernel function. The PCA is applied on the non-linearly mapped 
%  data. The result is a model describing non-linear data projection.
%
% Input:
%  X [num_data x dim] Training data.
%  
%  options [struct] Decribes kernel and output dimension:
%   .ker [string] Kernel identifier (see 'help kernel'); 
%     (default 'linear').
%   .arg [1 x narg] kernel argument; (default 1).
%   .new_dim [1x1] Output dimension (number of used principal 
%     components); (default num_data).
%
% Output:
%  Z [num x new_dim]            matrix of projections of X
%  model                        [structure] see below
%     .DR_method='Kpca';
%     .options                  [Structure]. Parameter for the Kpca of used model
%             .ker [string]     Kernel identifier (see 'help kernel').
%             .arg [1 x nargs]  Kernel argument(s).
%             .new_dim [1 x 1]  Output dimension (number of used principal
%     .X = X;                   [num_data x dim] original data 
%     .K=K;                     [num_data x num_data] Kernel matrix of X
%     .eigenvalues=Lambda;      all eigenvalue in vector form
%     .cputime = cputime - start_time;    % Time used for the process
%     .b [new_dim x 1]          Bias.
%     .Alpha [num x new_dim]    Multipliers.
%     
%      for FullRec
%     .eigenval=diag(Lambda);                           %Full U Eigenvectors     
%     .kercnt = num*(num+1)/2;
%     .MsErr = triu(ones(num,num),1)*model.eigval/num;  %MSE with respect to used basis vectors;
%     .mse = model.MsErr(options.new_dim);              %Mean square representation error of maped data   
%
% Example:
%  X = gencircledata([1;1],5,250,1);
%  model = kpca( X, struct('ker','rbf','arg',4,'new_dim',2));
%  XR = kpcarec( X, model );
%  figure; 
%  ppatterns( X ); ppatterns( XR, '+r' );
%  
% See also 
%  Kernel, Lpca.
%   
% About: 
%  Modification
%  WeiX, Jun 27th 2013, First Edition
%  WeiX, Dec 1st 2014, MK-II
%  WeiX, 4-1-2016, Add kAuto function for Gaussian kernel
%
% Reference:
% Statistical Pattern Recognition Toolbox 1999-2003, Written by Vojtech Franc and Vaclav Hlavac

%% Initialization and Parameters
% timer
start_time = cputime;
[num,dim] = size(X);  

% process input arguments
%-----------------------------------
if nargin < 2, options = [];end
if ~isfield(options,'ker'), options.ker = 'linear'; end
if ~isfield(options,'arg'), options.kAuto=1; end
if ~isfield(options,'new_dim'), options.new_dim = num; end
if ~isfield(options,'FullRec'), options.FullRec = 0; end      
if ~isfield(options,'kAuto'),  options.kAuto=0; end                      % Default no auto para

r = options.new_dim;

%% Main
%Kernel parameter auto generate
if options.kAuto==1
    switch options.ker
        case 'gaussian'      
            Distance =pdist2(X,X,'euclidean');
            options.arg=sum(Distance(:).^2)/(num^2);   
            options.arg=sqrt(options.arg/2);
        case 'linear'      
                % no parameter is required for linear kernel
        otherwise
            error('Error: kAuto not supported for such kernel type.');    
    end
end

% compute kernel matrix
K = Kernel(X,options.ker,options.arg);
% % % model.K=K;
% Centering kernel matrix (non-linearly mapped data). double centering
J = ones(num,num)/num;
Kc = K - J*K - K*J + J*K*J;     %This is actually double centering. 

% eigen decomposition of the kernel marix
% [U,D] = eig(Kc);			% U is the a in paper

if r >= 1   %calculated first r bases.
    [U,D] = eigs(Kc, r);			% U is the a in paper
%     [u,s,v] = svds(y,r);
    
    eigenValue = diag(D);      
    [eigenValue,order]=sort(-eigenValue);    
    eigenValue=-eigenValue;
    U=U(:,order);  
    
    eigenValue = isreal(eigenValue)*eigenValue;
    U = U * diag(isreal(eigenValue));
    
    rank = r;
    CumuEnergy = cumsum(eigenValue)./sum(eigenValue);
        
elseif r<1   
    [U,D] = eig(Kc);			% U is the a in paper
    
    eigenValue = diag(D);      
    [eigenValue,order]=sort(-eigenValue);    
    eigenValue=-eigenValue;
    U=U(:,order);  
    
    eigenValue = isreal(eigenValue)*eigenValue;
    U = U * diag(isreal(eigenValue));
    
    CumuEnergy = cumsum(eigenValue)./sum(eigenValue);
    idx = find(CumuEnergy >=r);
    rank = idx(1); 
end

options.new_dim = rank;
% [U,D] = eigs(Kc,options.new_dim);			% U is the a in paper
% Lambda=real(diag(D));
% U=real(U);

% normalization of eigenvectors to be orthonormal 
% for k = 1:num,
Lambda = eigenValue;

for k = 1:options.new_dim
  if Lambda(k) > (0 + eps)
     U(:,k)=U(:,k) / (sqrt(Lambda(k))+eps); 
  else
     U(:,k)=U(:,k) * 0; 
  end
end

% Sort the eigenvalues and the eigenvectors in descending order.
% [Lambda,order]=sort(-Lambda);    
% Lambda=-Lambda;
% U=U(:,order);               		%U becomes the a_wave matrix in paper    

% use first new_dim principal components
A=U(:,1:rank);              

% compute Alpha and compute bias (implicite centering)
% of kernel projection
model.Alpha = (eye(num,num)-J)*A;
Jt=ones(num,1)/num;

%model.b = A'*(J'*K*Jt-K*Jt); 
% This can also be written as 
model.b=(-1/num)*A'*(eye(num,num)-J)*K*ones(num,1);% require check

z=model.Alpha'*K+model.b*ones(1,num); % require check
z=z';

z = real(z);

%% Recording & Output
model.rank = rank;

model.Z=z;

model.DR_method='Kpca';
model.options = options;
model.X = X;

model.K=K;

model.eigenvalues=Lambda;
model.cputime = cputime - start_time;

% Full Information. Conditional output. (For Further Research without recalculation. Mass Memoey required)
if options.FullRec == 1   
    model.Kc=Kc;
    
    model.eigenval=diag(Lambda);              % Full U Eigenvectors     
    model.kercnt = num*(num+1)/2;
    model.MsErr = triu(ones(num,num),1)*model.eigval/num;
    model.mse = model.MsErr(options.new_dim);
    model.cputime = cputime - start_time;     % Update process time
end

return;
