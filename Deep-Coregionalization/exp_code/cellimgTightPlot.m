function cellimgTightPlot(cellImg,cRange)
% cellImg is a 2d cell. each element is an image 
% assert(length(nameList) == length(Y), 'Y and nameList not match')
% nExample = length(id);
% nMethod = length(Y);

yLabelList ={'t=1s','t=2s','t=3s','t=4s','t=5s'};

[nRow, nCol] = size(cellImg);
%% plot the colorbar 
figure()
set(gcf,'Position',[1 1 1000 70])
colormap hot
caxis(cRange);
axis off
h = colorbar;
h.Location = 'north'
set(h,'Position',[0.05 0.05 0.9 0.5]);
h.FontSize = 20;
h.TickLabels = h.Ticks

%% main plot
figure()
set(gcf,'Position',[1 1 200*nCol 200*nRow])
% set(gca,'position',[0 0 1 1],'units','normalized')
clf
[ha, pos] = tight_subplot(nRow, nCol, [.01 .01],[.15 .01],[.01 .03])

for i = 1:nRow
    for j = 1:nCol       
        axes(ha(i + (j-1)*nCol ));  
        pcolor(cellImg{i,j});
        
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

%% editing on the right column
for i = 1:nRow
    for j = nCol     
        axes(ha(i + (j-1)*nCol ));  
        caxis(cRange);
    end
end

%% editing on the bottom taw
for i = nRow
    for j = 1:nCol       
        axes(ha(i + (j-1)*nCol ));  
        
    end
end

%% editing on the left column
for i = 1:nRow
    for j = 1      
        axes(ha(i + (j-1)*nCol ));  
        
    end
end

%% editing on the top

for i = 1
    for j = 1:nCol       
        axes(ha(i + (j-1)*nCol ));  
        
    end
end


end