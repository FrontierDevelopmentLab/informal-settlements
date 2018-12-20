function [c,h] = plotTreebaggerDecisionSurface(Bag,x1Lims,x2Lims,nxRes,nyRes)
% Plots the decision surface from the output of TreeBagger, see
% plotCCFDecisionSurface - call structure is the same except first input is
% the output of TreeBagger instead of genCCF.

if ~exist('nxRes','var')
    nxRes = 300;
end

if ~exist('nyRes','var')
    nyRes = 300;
end

if ~exist('n_contours_or_vals','var')
    n_contours_or_vals = [];
end

[x1,x2] = meshgrid(linspace(x1Lims(1),x1Lims(2),nxRes),linspace(x2Lims(1),x2Lims(2),nyRes));
preds = predict(Bag,[x1(:),x2(:)]);
[c,h] = plotDecisionSurface(x1,x2,preds,n_contours_or_vals);

end

