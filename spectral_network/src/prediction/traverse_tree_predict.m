function [leaf_mean, leaf_node] = traverse_tree_predict(tree,X)
%[leaf_mean, leaf_node] = traverse_tree_predict(tree,X)
%
% Traverses the tree to get a prediction.  Splits X to left and right child
% then recursively calls self using each partition and the corresponding
% left and right sub tree.  This continues until called on a leaf, where it
% returns the mean of the leaf and, if requested, the full details of the
% corresponding leaf node.  These are then returned as array with the same
% number of rows as X.

if tree.bLeaf
    
    leaf_mean = bsxfun(@times,tree.mean,ones(size(X,1),1));
    if nargout>1
        leaf_node = repmat({tree},size(X,1),1);
    end
    
else
    if isfield(tree,'rotDetails') && ~isempty(tree.rotDetails)
        X = bsxfun(@minus,X,tree.rotDetails.muX)*tree.rotDetails.R;
    end
    
    if isfield(tree,'featureExpansion') && ~isempty(tree.featureExpansion)
        bLessChild = (tree.featureExpansion(X(:,tree.iIn))*tree.decisionProjection)<=tree.paritionPoint;
    else
        bLessChild = (X(:,tree.iIn)*tree.decisionProjection)<=tree.paritionPoint;
    end
    
    leaf_mean = NaN(size(X,1),numel(tree.mean));
    if nargout>1
        leaf_node = cell(size(X,1),1);
    end
    
    if any(bLessChild)
        if nargout>1
            [leaf_mean(bLessChild,:), leaf_node(bLessChild,:)] = traverse_tree_predict(tree.lessthanChild,X(bLessChild,:));
        else
            leaf_mean(bLessChild,:) = traverse_tree_predict(tree.lessthanChild,X(bLessChild,:));
        end
    end
    if any(~bLessChild)
        if nargout>1
            [leaf_mean(~bLessChild,:), leaf_node(~bLessChild,:)] = traverse_tree_predict(tree.greaterthanChild,X(~bLessChild,:));
        else
            leaf_mean(~bLessChild,:) = traverse_tree_predict(tree.greaterthanChild,X(~bLessChild,:));
        end
    end
end
end