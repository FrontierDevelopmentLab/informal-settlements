function Z = random_feature_expansion(X,w,b)
% Used to expand the random features in kernel CCA
Z = cos(bsxfun(@plus,X*w,b));
end