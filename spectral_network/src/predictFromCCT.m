function [leaf_mean, leaf_node] = predictFromCCT(tree,X)
%predictFromCCT predicts output using trained tree
%
% [leaf_mean, leaf_node] = predictFromCCT(tree,X)
%
% Inputs:   tree = output strcut from growTree
%              X = processed input features
% Outputs:  leaf_mean = Mean of outputs present at the leaf.  For
%                       classification then this represents the class
%                       probability, for regression it is simply the output
%                       mean.
%           leaf_node = The full leaf node details for the assigned point.
%                       WARNING: REQUESTING THIS WILL BE MUCH SLOWER THAN
%                       JUST TAKING ONE OUTPUT!
%
% 13/02/17

if isfield(tree,'inputProcessDetails')
    X = replicateInputProcess(X,tree.inputProcessDetails);    
end

% Any values left as NaN now need to be randomly assigned
X = random_missing_vals(X);

if nargout>1
    [leaf_mean, leaf_node] = traverse_tree_predict(tree,X);
else
    leaf_mean = traverse_tree_predict(tree,X);
end