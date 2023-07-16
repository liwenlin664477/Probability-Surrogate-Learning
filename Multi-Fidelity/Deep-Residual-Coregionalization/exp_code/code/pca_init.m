function [z, model] = pca_init(y,r)
% y must be [N x d] matrix, N being the number of sample

N = size(y,1);

y_mean = mean(y);
y = y - repmat(y_mean,N,1);

[U,S,V] = svds(y,r);

% z = U * S;
% V = V';
% Vinv = V; 

z = U;
Vinv = V * inv(S); 
V = S * V';



model.V = V;
model.Vinv = Vinv;
model.mean = y_mean;


end