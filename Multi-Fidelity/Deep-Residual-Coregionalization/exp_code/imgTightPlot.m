function imgTightPlot(Y,id,nameList,cRange)
% id = 1:6;
% cRange = [0,1];
% nameList = {'PCA-GP-F1','PCA-GP-F2','HOGP-F1','HOGP-F2',...
%     'ISOMAP-GP-F1','ISOMAP-GP-F2','KPCA-GP-F1','KPCA-GP-F2','MF-HOGP','Truth'};

assert(length(nameList) == length(Y), 'Y and nameList not match')

nExample = length(id);
nMethod = length(Y);

%% plot the colorbar 
figure()
% set(gca,'OuterPosition',[0,100,300,300])
set(gcf,'Position',[1 1 1000 70])
colormap hot
caxis(cRange);
axis off
h = colorbar;
h.Location = 'north'
set(h,'Position',[0.05 0.05 0.9 0.5]);
h.FontSize = 20;
h.TickLabels = h.Ticks
% set(h,'Ticks',linspace(cRange(1),cRange(2),10))
% set(fig1,'OuterPosition',pos) 
% print(['2lv_instHeatMapBar_',mat2str(nTr),'_',dataName],'-depsc','-opengl')




%% main plot
figure()
% set(gca,'OuterPosition',[0,100,300,300])
set(gcf,'Position',[1 1 200*nMethod 200*nExample])
% set(gca,'position',[0 0 1 1],'units','normalized')
clf
[ha, pos] = tight_subplot(nExample, nMethod, [.01 .01],[.15 .01],[.01 .03])

for i = 1:nMethod
    for j = 1:nExample
        
        axes(ha(i + (j-1)*nMethod ));
%         pcolor(reshape(yplot{i}(j,:),100,100))
        pcolor(squeeze(Y{i}(j,:,:)));
        
        shading interp
        colormap hot
        caxis(cRange);
    %     colorbar

    %     legend(num2str(id(:)))
        ax = gca;
        ax.FontSize = 15;
    %     ax.XLim = [1,length(nTr_lv2_list)];
        ax.XTick = [];
        ax.YTick = [];
%         xlabel(name_list{i})
    end
    
end

%% adding on the right column
i=nMethod;
for j = 1:nExample
    caxis(cRange);
%     colorbar
end
    

%% adding on the bottom taw
for i = 1:nMethod
    for j = nExample
        axes(ha(i + (j-1)*nMethod ));      
        xlabel(nameList{i})
    end
    
end




end