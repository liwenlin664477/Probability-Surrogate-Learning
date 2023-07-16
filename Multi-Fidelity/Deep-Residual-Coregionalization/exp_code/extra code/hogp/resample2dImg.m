%resample2dImg
% for 2d image only

clear 

dataName = 'LBracket01';
% dataName = 'canti';
%dataName = 'canti_10x10';
% dataName = 'gaussImg_v1_03';
% dataName = 'ns_v1_03';
% dataName = 'ns_v1_04';

load(dataName);

[nCoor1,nCoor2,nSample] = size(y);

coor1 = 1:nCoor1;
coor2 = 1:nCoor2;
% meshgrid(x,y)
[id1,id2] = ndgrid(1:nCoor1,1:nCoor2);
% [id1,id2] = meshgrid(1:nCoor1,1:nCoor2);
y_pertub = zeros(nCoor1-1,nCoor2-1,nSample);

rng(1)
for i = 1:nSample
    
    perturb1 = rand(size(id1)-1) * 1;
    perturb2 = rand(size(id2)-1) * 1;

    pertubId1= id1(1:end-1, 1:end-1) + perturb1;
    pertubId2= id2(1:end-1, 1:end-1) + perturb2;
    
%     pertubY(:,:,i) = interp2(id1,id2,y(:,:,i),pertubId1,pertubId2);
    f = griddedInterpolant(id1,id2,y(:,:,i), 'spline');
    y_pertub(:,:,i) = f(pertubId1,pertubId2);
%     pertubY(:,:,i) = griddedInterpolant(id1,id2,y(:,:,i),pertubId1,pertubId2);
    
    
    % visualize
%     figure(1)
%     subplot(1,2,1)
%     pcolor(y(:,:,i))
%     shading interp
%     subplot(1,2,2)
%     pcolor(y_pertub(:,:,i))
%     shading interp
%     
%     if floor(10*i/nSample)>floor(10*(i-1)/nSample), fprintf('.'), end
        
end

%% save
save(dataName,'y_pertub','-append')



