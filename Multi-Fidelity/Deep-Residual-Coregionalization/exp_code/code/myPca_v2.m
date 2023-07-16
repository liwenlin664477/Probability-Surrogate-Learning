function [V,Z_tr,Z_te] = myPca_v2(Y_tr,Y_te,r)
    
    [U,S,V] = svds(Y_tr,r);
    
    Z_tr = U*S;
    V = V';
    
    Z_te = Y_te * V';

end