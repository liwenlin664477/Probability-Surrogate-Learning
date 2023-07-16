% exp_v2_03
% logg: v1_01: for 2 layer experiment. run specific method 
%       v1_02: run all method
%       v1_03: all kpca, isomap comparison
%       v2_01: new archetecture and data saving
%       v2_03: all test method
clear
rng(1234)

nlv = 2;
% addpath(genpath('./../../code'))
dataName = 'uniform_10x10';
load(dataName)

x=X;
Y_lv1 = permute(Y_lv1,[2,3,1]);
Y{1} = reshape(Y_lv1,[],size(Y_lv1,3))';
Y{2} = reshape(Y_lv2,size(Y_lv2,1),[]);

% Y = [];
% for k = 1:nlv
%     Y{k} = reshape(Y_lv1{k},size(Y_lv1{k},1),[]);
% end

k=1;
funcList{k} = @mlvPcaGp; k=k+1;
funcList{k} = @mlvIsomapGp; k=k+1;
funcList{k} = @mlvKpcaGp; k=k+1;
funcList{k} = @mlvHogp; k=k+1;
funcList{k} = @deepPcaGpx12; k=k+1;

%% prepare test data
% nte = 64;
idte = 1:64;

xte = x(idte,:);    
x(idte,:) = [];     %remove test data

yte = Y{nlv}(idte,:);
for k = 1:nlv
    Y{k}(idte,:) = [];    %remove test data
end
%% experiment setting
nTr_lv2_list = [16,32,64];
% nTr_lv2_list = [128];
r_list = 5:5:20;

%% main
errRec = [];
for i = 1 : length(nTr_lv2_list)
    %update tr te data
    nTr_lv2 = nTr_lv2_list(i);
    xtr = x;
    Ytr = Y;
    Ytr{2} = Y{2}(1:nTr_lv2,:);
    
    for j = 1:length(r_list)
        r = r_list(j);
        
        for k = 1:length(funcList)
            func = funcList{k};
            try
                Ypred = [];
                [Ypred, model] = func(xtr,Ytr,xte,r);
%                 errRec = errRec_write(errRec, Ypred, yte, i,j,k);
                err = err_cell_eval(Ypred,yte);
                
                % save result
%                 mfilename
%                 mkdir(mfilename);
                saveFolder = [mfilename,'/',dataName,'/',func2str(func)];
                mkdir(saveFolder);
                saveName = ['lv2ntr',num2str(nTr_lv2),'_r',num2str(r)];
                saveName = [saveFolder,'/',saveName];
                save(saveName,'err');
            end
        end 
    end
    
end
%%
% mkdir(mfilename);
% saveFolder = [mfilename,'/',dataName];
% save(saveFolder,'errRec','nTr_lv2_list','r_list');

%% plot
kIndexs = repmat("MFFGP",12,1);
kIndexs(13) = "PCA-GP";
kIndexs(14) = "HOGP";
kIndexs(15) = "Isomap-GP";
kIndexs(16) = "Kpca-GP";

[mat,xlabels,ylabels] = errRec2matrix_v1_1(errRec,nTr_lv2_list,r_list,kIndexs);
[mat2,xlabels2,ylabels2] = errRec2matrix_v1_2(errRec,nTr_lv2_list,r_list,kIndexs);

mat.mse(mat.mse==0) = nan;
mat2.mse(mat2.mse==0) = nan;

showId = [4,13,15,16];
figure(1)
clf
plot([1:length(ylabels)], mat.mse(:,showId))
legend(kIndexs(showId));
% xticks('manual')
ax = gca;
ax.XTickLabelMode = 'manual';
ax.XTick = [1:1:length(ylabels)];
ax.XTickLabel = ylabels;
ax.XTickLabelRotation=45;
xlabel('r; Ntrain');
ylabel('MSE');
ax.FontSize = 12;

 
% xticks([1:1:length(ylabels)]);
% xticklabels(ylabels)

figure(2)
clf
plot([1:length(ylabels2)], mat2.mse(:,showId))
legend(kIndexs(showId));

ax = gca;
ax.XTickLabelMode = 'manual';
ax.XTick = [1:1:length(ylabels2)];
ax.XTickLabel = ylabels2;
ax.XTickLabelRotation=45;
xlabel('Ntrain; r');
ylabel('MSE');
ax.FontSize = 12;
%% 



%%
function err = err_cell_eval(yPred,yTrue)

    assert(iscell(yPred),'yPred is not a cel');
    err = [];
    for k = 1:length(yPred)
        err_k = err_eval(yPred{k},yTrue);
        err = cat(2,err,err_k);
    end

end

function err = err_eval(yPred,yTrue)
        
    err2 = mean((yPred - yTrue).^2, 2);   
    err.mse_sample = err2;
    err.mse_dims = mean((yPred - yTrue).^2, 1); 
    
    err.mse = mean(err2);
    err.mse_std = std(err2);

    re = err2 ./ (mean(yTrue.^2, 2)+eps);
    err.re = mean(re);
    err.re_std = std(re);
    
    yMean = mean(yTrue(:));
    yStd = std(yTrue(:));
    
    %normalize error
    yPred = (yPred - yMean)/yStd;
    yTrue = (yTrue - yMean)/yStd;
    
    nerr2 = mean((yPred - yTrue).^2, 2);
    err.nmse = mean(nerr2);
    err.nmse_std = std(nerr2);

end


function [mat,xlabels,ylabels] = errRec2matrix_v1_1(errRec,iIndexs,jIndexs,kIndexs)
    mat = structfun(@(x)reshape(x,[],length(kIndexs)), errRec,'UniformOutput',false);
    xlabels = kIndexs;
    [jG,iG] = meshgrid(jIndexs,iIndexs);
    ylabels = [jG(:),iG(:)];
    ylabels = num2str(ylabels);
    ylabels = string(ylabels);
    
%     mat(mat==0) = nan;
end


function [mat,xlabels,ylabels] = errRec2matrix_v1_2(errRec,iIndexs,jIndexs,kIndexs)
    mat = structfun(@(x)reshape(permute(x,[2,1,3]),[],length(kIndexs)), errRec,'UniformOutput',false);
%     mat = structfun(@(x) x(x==0)=nan, errRec,'UniformOutput',false);
    xlabels = kIndexs;
    [iG,jG] = meshgrid(iIndexs,jIndexs);
    ylabels = [iG(:),jG(:)];
    ylabels = num2str(ylabels);
    ylabels = string(ylabels);
     
end

%%
