% cigp_cat_DEMO
clear 

n = 128;
d = 8;

id = randperm(n);
idtr = sort(id(1:n/2));
idte = sort(id(n/2+1:end));

% x = rand(n,d)*diag([d:-1:1]);
% y = sum(x(:,1:5:16).^2,2);

x = rand(n,d);
x_scale = x*diag([d:-1:1]);
y = sum(x_scale(:,2:2:8).^2,2);

[y,idx] = sort(y);
x = x(idx,:);

% main
i=1;
xtr = x(idtr,1:i);
ytr = y(idtr,:);

xte = x(idte,1:i);
yte = y(idte,:);
model1{i} = cigp_v2_03(xtr, ytr, xte);
yPred(:,i) = model1{i}.yTe_pred;
yPred2(:,i) = model1{i}.yTe_pred;
model2{i} = model1{i};

for i = 2:d
    xtr = x(idtr,1:i);
    ytr = y(idtr,:);
    
    xte = x(idte,1:i);
    yte = y(idte,:);

    model1{i} = cigp_v2_03(xtr, ytr, xte);
    yPred(:,i) = model1{i}.yTe_pred;
    
%     model2{i} = cigp_cat(model1{i-1}, xtr, ytr, xte);
    model2{i} = cigp_cat_v03(model2{i-1}, xtr, ytr, xte);
    yPred2(:,i) = model1{i}.yTe_pred;
    
end
figure(1) 
plot(yte,'ok-')
hold on 
plot(yPred,'+b-')
plot(yPred2,'^r-')
hold off

figure(2) 
plot(yte,'ok-')
hold on 
plot(yPred(:,d),'+b-')
plot(yPred2(:,d),'^r-')
hold off



  
