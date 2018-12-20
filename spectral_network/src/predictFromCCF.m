function [forestPredicts, forestProbs, treeOutputs] = predictFromCCF(CCF,X)
%predictFromCCF predicts outputs using trained forest
%
% [forestPredicts, forestProbs, treePredicts] = predictFromCCF(CCF,X)
%
% Inputs:   CCF = output from genCCF.  This is a structure with a field 
%                 Trees, giving a cell array of tree structures, and 
%                 options which is an object of type optionsClassCCF
%           X = input features at which to make predictions, each row 
%               should be a seperate data point
% Outputs:     
%   forestPredicts = Array of numeric predictions corresponding
%               to the prediction.  
%                   - For regression this is a vector of scalar predictions
%                   - For multivariate regression it is a matrix were each 
%                     column is a different output variable.
%                   - For classification where Y was original provided as
%                     a numeric array then it is a vector predicted class 
%                     labels
%                   - For classification where Y was originally a logical
%                     array or cell array of strings then is a vector of
%                     indices to CCF.classNames
%                   - For multiple output classification then it is a
%                     matrix where columns correspond to the output of the
%                     appropriate single output classification case.
%  forestProbs = Assigned probability to each class for classification.  
%                Empty for regression.  Columns are different indexes to
%                CCF.classNames
%  treeOutputs = Individual tree outputs stored as a NxLxK array where N is
%                the number of predicted data points, L is the number of
%                trees and K is the number of predictions.  K=1 for
%                regression, K = number of classes for classification, and
%                for regression then each output is concatenated in the
%                third dimension.
%
% 23/07/17

X = replicateInputProcess(X,CCF.inputProcessDetails);

nTrees = numel(CCF.Trees);
% Preallocate output space
treeOutputs = repmat(reshape(predictFromCCT(CCF.Trees{1},X),size(X,1),1,[]),1,nTrees,1);
if CCF.options.bUseParallel
    Trees = CCF.Trees;
    parfor n=2:nTrees
        treeOutputs(:,n,:) = predictFromCCT(Trees{n},X);
    end
else
    for n=2:nTrees
        treeOutputs(:,n,:) = predictFromCCT(CCF.Trees{n},X);
    end
end

[forestPredicts, forestProbs] = treeOutputsToForestPredicts(CCF,treeOutputs);

end