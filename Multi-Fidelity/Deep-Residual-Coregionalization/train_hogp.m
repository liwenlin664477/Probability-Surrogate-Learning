function [] = train_hogp(rank, data_name, res_name)
    
    addpath(genpath('exp_code'));
    addpath(genpath('progress'));

    data = load(data_name);

    X_train = data.X_train;
    Y_train = data.Y_train;
    
    X_test = data.X_test;
    Y_test = data.Y_test;
    
    %fprintf('train\n')

    model = hogp_v2(X_train, tensor(Y_train), X_test, rank);
    ystar = model.yPred.data;
    
    %fprintf('done train\n')
    
%     size(X_test)
%     size(ystar)

    save(res_name, 'ystar')
    
    %fprintf('done hogp\n')

end

