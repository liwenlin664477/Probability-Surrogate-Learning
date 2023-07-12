function [V,Z] = myPca(Y,r)
    
    [U,S,V] = svds(Y,r);
    Z = U*S;
    V = V';

end