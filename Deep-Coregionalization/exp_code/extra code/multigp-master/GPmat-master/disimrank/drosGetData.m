function [y, yvar, gene, times, scale, rawExp, rawVar] = drosGetData(drosexp, genes, getmedians),

% DROSGETDATA Get Drosophila data as processed by mmgMOS.
% FORMAT
% DESC Extract given genes from drosexp structure.
% ARG drosexp : drosexp structure as returned by drosLoadData
% ARG genes : indices or gene labels to extract
% ARG getmedians : return median values and zero variances
% RETURN y : the normalised expression levels.
% RETURN yvar : the variance of the normalised expression levels.
% RETURN gene : the gene names and Affymetrix array tags.
% RETURN times : the times of the expression measurements.
% RETURN scale : the scaling factor applied to normalise.
% RETURN rawExp : the raw gene expresion level.
% RETURN rawVar : the raw variance of the gene expression.
% 
% SEEALSO : drosLoadData
%
% COPYRIGHT : Neil D. Lawrence, 2006
% COPYRIGHT : Antti Honkela, 2007

% SHEFFIELDML

if nargin < 3,
  getmedians = 0;
end

if iscell(genes),
  genes = drosFindGeneinds(drosexp, genes, 0, 1);
end

N = length(genes);

gene = drosexp.genes(genes);
if size(gene, 1) < size(gene, 2),
  gene = gene';
end
gene = [gene, gene];

rawExp = zeros(36, N);
rawVar = zeros(36, N);
yFull = zeros(36, N);
yFullVar = zeros(36, N);

rawExp = drosexp.mean(genes, :)';
rawVar = (drosexp.se(genes, :)').^2;
if getmedians,
  yFull = exp(drosexp.pctiles(genes, :, 3))';
  yFullVar = zeros(size(yFull));
else
  if isfield(drosexp, 'fitmean'),
    yFull = drosexp.fitmean(genes, :)';
    yFullVar = drosexp.fitvar(genes, :)';
  else
    for k=1:N,
      prof = squeeze(drosexp.pctiles(genes(k), :, :));
      for l=1:36,
	t = distfit(exp(prof(l, :)), 'normal');
	yFull(l, k) = t(1);
	yFullVar(l, k) = t(2) .^ 2;
      end
    end
  end
end

% Rescale so that average standard deviation of curves is 1.
scale = sqrt(var(yFull));
scaleMat = ones(size(yFull, 1), 1)*scale;
yFull = yFull./scaleMat;
yFullVar = yFullVar./(scaleMat.*scaleMat);

y{1} = yFull(1:12, :);
y{2} = yFull(13:24, :);
y{3} = yFull(25:36, :);
yvar{1} = yFullVar(1:12, :);
yvar{2} = yFullVar(13:24, :);
yvar{3} = yFullVar(25:36, :);
times = (1:12)';
%save('./data/barencoData.mat', 'y', 'yvar', 'gene', 'times', 'scale', 'rawVar', 'rawExp');
