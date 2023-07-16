function [xtr,Ytr,xte,yte] = load_data_normal(dataName,nlv)
% load consistent tr te data. 
% may need Modify for different raw data

    load(dataName);
%     Y = [];
    Ytr = [];
    yte = [];
    
%     Ytr_interp = Ytr;
%     Yte_interp = Yte;
    
    ymean =  mean2(cell2mat(cat(1,Ytr_interp',Yte_interp')));
    ystd =  std2(cell2mat(cat(1,Ytr_interp',Yte_interp')));
    
    for k = 1:nlv
        Ytr{k} = reshape(Ytr_interp{k},size(Ytr_interp{k},1),[]);
        Ytr{k} = (Ytr{k} - ymean) ./ ystd;
        
        Yte{k} = reshape(Yte_interp{k},size(Yte_interp{k},1),[]);
        Yte{k} = (Yte{k} - ymean) ./ ystd;
    end 
    yte = Yte{nlv};
    
    xmean = mean(xtr);
    xstd = std(xtr);
    ntr = size(xtr,1);
    xtr = (xtr - repmat(xmean,ntr,1)) ./ repmat(xstd,ntr,1);
    nte = size(xte,1);
    xte = (xte - repmat(xmean,nte,1)) ./ repmat(xstd,nte,1);
    
end