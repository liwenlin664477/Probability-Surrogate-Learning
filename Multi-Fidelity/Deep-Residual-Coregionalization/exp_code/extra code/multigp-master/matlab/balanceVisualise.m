function handle = balanceVisualise(channels, skel, padding)

% BALANCEVISUALISE Draws a skel representation of 3-D data balance motion
% FORMAT
% DESC draws a skeleton representation in a 3-D plot.
% ARG channels : the channels to update the skeleton with.
% ARG skel : the skeleton structure.
% ARG padding : a vector with positions to fill the channel (default zeros) 
% RETURN handle : a vector of handles to the plotted structure.
%
% SEEALSO : balanceModify
%
% COPYRIGHT : Mauricio Alvarez, Neil D. Lawrence, 2009  

% MULTIGP

global ffPos
global rotMat
if nargin<3
    padding = 0;
end
channels = [channels zeros(1, padding)];
vals = skel2xyz(skel, channels);
connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);


vals = vals - repmat(vals(4,:), size(vals,1), 1);
ffPos = vals(5, :)';
% thet = acos(ffPos(2)/sqrt(ffPos(2:3)'*ffPos(2:3)));
% rotMat2 = rotationMatrix(thet, 0, 0, 'x');
% ffPos = rotMat2*ffPos;
thet = asin(ffPos(1)/sqrt(ffPos(1:2)'*ffPos(1:2)));
rotMat = rotationMatrix(0, thet, 0, 'y');
vals = vals*rotMat';
ffPos = rotMat*ffPos;
handle(1) = plot3(vals(:, 1), vals(:, 3), vals(:, 2), '.');
axis ij % make sure the left is on the left.
set(handle(1), 'markersize', 20);
%/~
%set(handle(1), 'visible', 'off')
%~/
hold on
grid on
for i = 1:length(indices)
  handle(i+1) = line([vals(I(i), 1) vals(J(i), 1)], ...
              [vals(I(i), 3) vals(J(i), 3)], ...
              [vals(I(i), 2) vals(J(i), 2)]);
  set(handle(i+1), 'linewidth', 2);
end
axis equal
xlabel('x')
ylabel('z')
zlabel('y')
axis on