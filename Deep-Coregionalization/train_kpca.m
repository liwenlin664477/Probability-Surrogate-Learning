function [] = train_kpca(rank, data_name, res_name)
    
    addpath(genpath('exp_code'));

    data = load(data_name);

    X_train = data.X_train;
    Y_train = data.Y_train;
    
    X_test = data.X_test;
    Y_test = data.Y_test;

    [ystar, ~] = kpcaGp_v01(X_train, Y_train, X_test, rank);
    
%     size(X_test)
%     size(ystar)

    save(res_name, 'ystar')

end

