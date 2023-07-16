function [Z,Model] = mfPca(Y, r, iMethod)
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
        [z, model] = pcaInit(yCat, r);
        
        for k = 1:nLv 
            Z{k} = model.project(Y{k});
        end
         
%         U = repmat({U},1,nLv);
%         Ut = cellfun(@transpose,U, 'UniformOutput',false);
%         Z = cellfun(@mtimes,Y,Ut, 'UniformOutput',false);
        
    case 2 %independent combined basis
            error('current not in use')
%         for k = 1:nLv 
%             [U_ib{k},~] =  myPca(Y{k}, r);
%             
%             %ensure same direction
%             direct = sign(diag(U_ib{k}*U_ib{1}'));
%             U_ib{k} = diag(direct) * U_ib{k};
%             
%             U{k} = cell2mat(U_ib');
% %             Z{k} = Y{k} * U{k}';
%             Z{k} = U{k}' \ Y{k}';
%             Z{k} = Z{k}';
%         end
        
    case 3 % progressive basis
%         for k = 1:nLv 
%             [z1{k}, Model1{k}] = pcaInit(Y{k},r);
%         end
%         Model{1} = Model1{1};
%         for k = 2:nLv
%             Y_residual = Y{k} - Model{k-1}.recover(Model{k-1}.project(Y{k}));
%             [z1{k}, Model{k}] = pcaInit(Y_residual,r);
%         end
%         
%         U{1} = U_ib{1};
%         Z{1} = Z_ib{1}; 
%         for k = 2:nLv
% %             U_lastLv{k} = cell2mat(U_ib');
%             U_part1 = U{k-1};
% %             Z_part1 = U_part1' \ Y{k}';
%             Z_part1 = Y{k} * U_part1';
% %             sum(Z_part11 - Z_part1')
%             
%             Y_residual = Y{k} - Z_part1*U_part1;
%             [U_part2, Z_part2] =  myPca(Y_residual, r);  
%             U{k} = [U_part1;U_part2];
%             Z{k} = [Z_part1,Z_part2];   %!! DOUBLE CHECK
%         end
        error('current not in use')
        
    case 4 % residual independent basis

        [U{1}, Z{1}] =  myPca(Y{1}, r);  
        for k = 2:nLv 
            nSample = size(Y{k}, 1);
            Yr{k} = Y{k} - Y{k-1}(1:nSample,:);
%             [Ur{k}, Zr{k}] =  myPca(Yr{k}, r);
            
            % use normal U and Z to represent the 
%             U = Ur;
%             Z = Zr;
            [U{k}, Z{k}] =  myPca(Yr{k}, r);
        end

    otherwise 
        error('no method')
        
        
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