function YpermPreds = randperm_preds(tree, X, bOutOfBag)
% Calculates D sets of predictions for a tree, each with column d of X 
% randomly permuted.  Currently only used by feature_importance function.
%
% Inputs: tree
%    X = Samples to test
%    bOutOfBag = Use only the out of bag indices, requires CCF-Bag to
%         have been used in the first place.
%
% Outputs: YpermPreds = predicts
%
% Tom Rainforth 29/06/2017

if ~exist('bOutOfBag','var')
    bOutOfBag = true;
end

if bOutOfBag
    X = X(tree.iOutOfBag,:);
end

% Any values left as NaN now need to be randomly assigned
X = random_missing_vals(X);
[N,D] = size(X);
YpermPreds = cell(D,1);

for d=1:D
    Xd_true = X(:,d);
    X(:,d) = X(randperm(N),d);
    YpermPreds{d} = predictFromCCT(tree,X);
    X(:,d) = Xd_true;
end

YpermPreds{D+1} = predictFromCCT(tree,X);

end


