function [c,h] = plotCCFDecisionSurface(CCF,x1Lims,x2Lims,nx1Res,nx2Res,n_contours_or_vals)
% Makes a contour plot for a decision surface produced by a CCF
%
% Inputs: CCF = Output from genCCF
%        x1lims,x2lims = Limits for input dimensions to plot between, e.g.
%                        [-1,1], [0,2]
%        nx1Res,nx2Res = Resolution of grid for each input, default 100
%        n_contours_or_vals = Number of contours or contour values to be
%                passed in.  Should be left blank for classification in
%                which case the contour is between each class.  For
%                regression then this is the 4th input passed to the
%                contourf function, default = 20.
%
% Outputs: [c,h] = Outputs from the contourf function

if ~exist('nxRes','var')
    nx1Res = 200;
end

if ~exist('nyRes','var')
    nx2Res = 200;
end

if ~exist('n_contours_or_vals','var')
    n_contours_or_vals = [];
end

[x1,x2] = meshgrid(linspace(x1Lims(1),x1Lims(2),nx1Res),linspace(x2Lims(1),x2Lims(2),nx2Res));
preds = predictFromCCF(CCF,[x1(:),x2(:)]);
[c,h] = plotDecisionSurface(x1,x2,preds,n_contours_or_vals);
end

