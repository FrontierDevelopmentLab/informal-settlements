function [c,h] = plotDecisionSurface(x1,x2,preds,n_contours_or_vals)
% Makes a contour plot for a decision surface given a grid of inputs and
% predictions
%
% Inputs: x1,x2 = Grids of corresponding input points corresponding to the
%                first and second dimension.  E.g. output from meshgrid
%        preds = Corresponding grid of predictions
%
% Outputs: [c,h] = Outputs from the contourf function

    uniquePreds = unique(preds);
    nVals = numel(uniquePreds);
    numericPreds = NaN(nVals,1);
    
    if iscell(uniquePreds)
        for n=1:nVals
            numericPreds(strcmpi(uniquePreds{n},preds)) = n;
        end
    else
        numericPreds = preds;
    end
    
    numericPreds = reshape(numericPreds,size(x1));
    
    if ~exist('n_contours_or_vals','var') || isempty(n_contours_or_vals)
        if nVals >= size(preds,1)/2
            % Presumably regression
            n_contours_or_vals = 50;
        else
            n_contours_or_vals = 1.5:1:(nVals-0.5);
        end
    end
    
    figure
    [c,h]= contourf(x1,x2,numericPreds,n_contours_or_vals);
    xlabel('x1')
    ylabel('x2')    
    axis square
    
end