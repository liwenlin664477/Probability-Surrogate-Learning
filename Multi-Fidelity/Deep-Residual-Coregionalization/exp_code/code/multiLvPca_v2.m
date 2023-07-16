function [Z,U] = multiLvPca_v2(Y, r, iMethod)
%using PCA to reduce each lv output dimension
assert(iscell(Y),'Y must be cell');

Z=[];
nLv = length(Y);
switch iMethod
    case 1  %ub uniform basis
%         for k = 1:nLv 
%             yCat = [yCat;ytr{k}];
%         end
        yCat = cell2mat(Y');
%         [U, ~] = myPca(yCat, r);
%         U = repmat({U},1,nLv);
%         [U,~] = myPca(yCat,r);
        [~,S,U] =  svds(yCat, r);
        
        Uinv = U * inv(S);
        U = S * U';
        
        U = repmat({U},1,nLv);
        Uinv = repmat({Uinv},1,nLv);
%         Ut = cellfun(@transpose,U,'UniformOutput',false);
        Z = cellfun(@mtimes,Y,Uinv,'UniformOutput',false);        
        
        
    case 2 %independent combined basis
        for k = 1:nLv 
            [U_ib{k},Z{k}] =  myPca(Y{k}, r);
            [Z_ib{k}, S_ib{k}, U_ib{k}] =  svds(Y{k}, r);  
            
            U_ib{k} = S_ib{k} * U_ib{k}';
            %ensure same direction
            direct = sign(diag(U_ib{k}*U_ib{1}'));
            U_ib{k} = diag(direct) * U_ib{k};
%             U{k} = U_ib{k};
            
            U{k} = cell2mat(U_ib');
%             Z{k} = Y{k} * U{k}';
            Z{k} = U{k}' \ Y{k}';
            Z{k} = Z{k}';
        end
        
    case 3 % progressive basis
        for k = 1:nLv 
%             [U_ib{k}, Z_ib{k}] =  myPca(Y{k}, r);              
            [Z_ib{k}, S_ib{k}, U_ib{k}] =  svds(Y{k}, r);              
        end
        
        U{1} = S_ib{1} * U_ib{1}';
        Uinv{1} = U_ib{1} * inv(S_ib{1});
        
        Z{1} = Z_ib{1}; 
        for k = 2:nLv
%             U_lastLv{k} = cell2mat(U_ib');
            Uinv_part1 = Uinv{k-1};
            U_part1 = U{k-1};
%             Z_part1 = U_part1' \ Y{k}';
            
            Z_part1 = Y{k} * Uinv_part1;
%             sum(Z_part11 - Z_part1')
            
            Y_residual = Y{k} - Z_part1*U_part1;
%             [U_part2, Z_part2] =  myPca(Y_residual, r);
            
            [Z_part2, S_part2, U_part2] =  svds(Y_residual, r);            
            Uinv_part2 = U_part2 * inv(S_part2);
            U_part2 = S_part2 * U_part2';
            
            U{k} = [U_part1;U_part2];
            Uinv{k} = [Uinv_part1,Uinv_part2];
            Z{k} = [Z_part1,Z_part2];   %!! DOUBLE CHECK
        end
        
    case 4 % residual independent basis

        [U{1}, Z{1}] =  myPca(Y{1}, r);  
        for k = 2:nLv 
            nSample = size(Y{k}, 1);
            Yr{k} = Y{k} - Y{k-1}(1:nSample,:);
%             [Ur{k}, Zr{k}] =  myPca(Yr{k}, r);
            
            % use normal U and Z to represent the 
%             U = Ur;
%             Z = Zr;
%             [U{k}, Z{k}] =  myPca(Yr{k}, r);
            [Z{k},S{k},U{k}] =  svds(Yr{k}, r);
            U{k} = S{k} * U{k}';
        end

    otherwise 
        error('no method')
        
        
end




end


function [V,Z] = myPca(Y,r)
    
    [U,S,V] = svds(Y,r);
%     Z = U*S;
%     V = V';
    
    Z = U;
    V = S*V';

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