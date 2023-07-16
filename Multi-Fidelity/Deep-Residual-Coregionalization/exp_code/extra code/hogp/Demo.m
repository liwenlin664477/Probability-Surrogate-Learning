% Demo

x = rand(64,1);
y = reshape(sinc(x(:) * x(:)'),64,8,8);


xtr = x(1:24,:);
ytr = y(1:24,:,:);

xte = x(25:end,:);
yte = y(25:end,:,:);

model = train_HOGP(xtr, tensor(ytr), xte, tensor(yte), 2, 0.001, 0.001);
% [pred_mean,model, pred_tr] = pred_HoGP(new_params, r, Xtr, ytr, Xtest, 'ard', 'linear');

