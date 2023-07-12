function [xtr,Ytr,xte,yte] = load_data(dataName,nlv)
% load consistent tr te data. 
% may need Modify for different raw data

    load(dataName);
    
%     Y = [];
    Ytr = [];
    yte = [];
    for k = 1:nlv
        Ytr{k} = reshape(Ytr_interp{k},size(Ytr_interp{k},1),[]);
%         Yte{k} = reshape(Ytr_interp{k},size(Ytr_interp{k},1),[]);
    end 
    yte = reshape(Yte_interp{nlv},size(Yte_interp{nlv},1),[]);
    
 
%         % get test data
%         idte = 1:128;
%         xte = x(idte,:);    
%         x(idte,:) = [];     %remove test data
% 
%         yte = Y{nlv}(idte,:);
%         for k = 1:nlv
%             Y{k}(idte,:) = [];    %remove test data
%         end
    

end