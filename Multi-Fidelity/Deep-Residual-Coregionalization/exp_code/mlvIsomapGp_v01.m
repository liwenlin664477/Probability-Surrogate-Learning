function [Ypred, model] = mlvIsomapGp_v01(xtr,Ytr,xte,r)
% use normal y(:)

    options.dim_new=r;                % New dimension
    options.neighborType='k';               % Type of neighbor.Choice:1)'k';Choice:2)'epsilon'
    options.neighborPara=10;                % parameter for choosing the neighbor. number of neighbor for "k" type and radius for 'epsilon'
    options.metric='euclidean';             % Method of measurement. Metric    
    
    preoptions.ReCoverNeighborType='k';     % Type of neighbor of new point. Choice:1)'k';Choice:2)'epsilon'
%     preoptions.ReCoverNeighborPara=10;      % Parameter of neighbor of new point
    

    for k = 1:length(Ytr)
        
        nSample = size(Ytr{k}, 1);
        kxtr = xtr(1:nSample,:);  
        
        if(r > nSample)
            options.dim_new = nSample;
        else 
            options.dim_new = r;
        end
        
        
        try
            [Ztr{k},isomapModel{k}] = Isomaps(Ytr{k},options);
%             [Ztr{k},isomapModel{k}] = isoma(Ytr{k},options);
    %         model{k} = cigp_v2(kxtr, Ztr{k}, xte);
            model_gp{k} = cigp_v2_03(kxtr, Ztr{k}, xte);

            Ypred{k} = Isomaps_PreImage(model_gp{k}.yTe_pred,isomapModel{k},preoptions);
%         train_pred = Isomaps_PreImage(utrain_pred(:,1:k),kpcaModel,preoptions);
        catch
            Ypred{k} = [];
        end
        
    end
    model.isomap = isomapModel;
    model.gp = model_gp;
end

%%
id = [1:8,14];
% errRec1 = reshape([errRec.mse], size(errRec));

name_list = {'PCA-GP-F1','PCA-GP-F2','HOGP-F1','HOGP-F2',...
    'ISOMAP-GP-F1','ISOMAP-GP-F2','KPCA-GP-F1','KPCA-GP-F2','MF-HOGP'};
lspec_list = {'bo-.','b>-','go-.','g>-','ko-.','k>-','mo-.','m>-','r^--'};
% lspec_list = {'ko-','bo-','kh-','bh-','ks-','bs-','k*-','b*-','r>-'};

figure(11)
% set(gca,'OuterPosition',[0,100,300,300])
set(gcf,'Position',[1 1 1000 400])
% set(gca,'position',[0 0 1 1],'units','normalized')
clf
[ha, pos] = tight_subplot(1, length(r_list), [.01 .04],[.15 .1],[.08 .02])
for ir = 1:length(r_list)
    
    plotline = squeeze(errRec1(:,ir,:));
%     plotline = log(plotline);
    axes(ha(ir));
    hold on 
    for j = 1:length(id)
        plot(plotline(:,id(j)), lspec_list{j}, 'LineWidth',2, 'MarkerSize',10);   
    end
    hold off
%     legend(num2str(id(:)))
    ax = gca;
    ax.FontSize = 20;
    ax.XLim = [1,length(nTr_lv2_list)];
    ax.XTick = 1:length(nTr_lv2_list);
    ax.XTickLabel = nTr_lv2_list;
    xlabel('Number of training for F2')
    title(['R=',num2str(r_list(ir))])
    
    ax.YLim =[0,0.3];
    
    grid on
%     ylim auto
%     ax.XTick = nTr_lv2_list;

%     plot_styles(plotline(:,id)  );
%     plot_styles(log(plotline(:,id))); 
end
% legend(num2str(id(:)))
% legend(name_list)
axes(ha(1));
ax = gca;
ax.YLim =[0,0.3];
ax.YTickLabelMode = 'auto';
ylabel('RMSE');

% print(['2lv_RMSE_',dataName],'-depsc')
% print(['2lv_NRMSE_',dataName,'_normal'],'-depsc')
%for original (non-normalized) data (exp1_v03)
print(['2lv_NRMSE_',dataName],'-depsc')




