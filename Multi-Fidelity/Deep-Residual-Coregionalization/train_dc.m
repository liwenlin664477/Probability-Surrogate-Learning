function [] = train_dc(rank, data_name, res_name)

    addpath(genpath('deep_coregional'));

    raw_mf_data = load(data_name);
    
    nfids = size(raw_mf_data.Xte_list, 2);

    for fid = 1:nfids
        ytr{fid} = raw_mf_data.ytr_list{fid};
    end

    Xtr = raw_mf_data.Xtr_list{1};

    raw_Xte = raw_mf_data.Xte_list{nfids};
    rank=20;
    Xte = raw_Xte;

    yte = raw_mf_data.yte_list{nfids};

    flag = false;

    while ~flag
        try
           [ypred, ~] = dc(Xtr, ytr, Xte, rank);
           flag=true;
        catch exception
           fprintf('### WARNING: Add rank by one rank=%d\n', rank)
           rank = rank + 1;
        end
    end

    nrmse = sqrt(mean((ypred-yte).^2))/sqrt(mean(yte.^2));

    save(res_name, 'ypred', 'nrmse');

end





