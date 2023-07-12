function yte = MFFGP_ub(r,xtr,ytr,xte)
% MFFGP uniform (combined) basis


nLv = length(ytr);
yCat=[];

%cat all training data
for k = 1:nLv 
    yCat = [yCat;ytr{k}];
end
[U, ~] = myPca(yCat, r);

%apply to each ytr{k}
for k = 1:nLv 
    ztr{k} = ytr{k} * U';
end

% last_zPred = [];
zte=[];
for k = 1:nLv 
    ntr = size(ytr{k},1);
    nte = size(xte,1);
    
    ixtr = xtr(1:ntr,:);
    last_iztr = lvCat(ztr,k-1, ntr);    
    ixtr = [ixtr,last_iztr];
    
    ixte = [xte,lvCat(zte,k-1 , nte)];
    
    model{k} = cigp_v2(ixtr, ztr{k}, ixte);
    zte{k} = model{k}.yTe_pred;
    yte{k} = zte{k} * U;
    
end


end

function ycat = lvCat(y,k,keepN)
%concatinate first k cell y into matrix, with preserved number of samples.

    if k == 0
        ycat = [];
    else
%         n = size(y{k});
        ycat = [];
        for i = 1:k
            ycat = [ycat,y{i}(1:keepN,:)];
        end
    end
end