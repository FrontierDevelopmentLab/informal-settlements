function [coeff,muX,vals] = pcaLite(X,bScale,bMakeFullRank)
% function [coeff,muX,vals] = pcaLite(X,bScale,bMakeFullRank)
%
% A faster and more natural pca function.  To transform data not in
% original sample X, use Z = bsxfun(@minus,Xnew,muX)*coeff where muX is the
% mean of the original sample.
%
% If bScale is true then the data is scaled according to the individual
% feature variances.  This is incorporated in the coeff matrix so that
% further transformations can still be done in manner given above.
%   Default = false
%
% If bMakeFullRank is true then the principle components are reduced to
% ensure that they are full rank.
%   Default = true;
%
% 21/05/15

if nargin<2 || isempty(bScale)
    bScale = false;
end

if nargin<3 || isempty(bMakeFullRank)
    bMakeFullRank = true;
end

muX = mean(X,1);
X = bsxfun(@minus,X,muX);
if bScale
    sig = std(X,[],1);
    X = bsxfun(@rdivide,X,sig);
end

[~,v,coeff] = svd(X,'econ');

if bMakeFullRank
    coeff = coeff(:,diag(v)>(eps*v(1)*max(size(X))));
end

if nargout>2
    vals = X*coeff;
end

if bScale
    coeff = bsxfun(@rdivide,coeff,sig');
end