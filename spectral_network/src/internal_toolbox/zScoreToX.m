function X = zScoreToX(zScore,mu_X,std_X)

X = bsxfun(@plus,bsxfun(@times,zScore,std_X),mu_X);
X(isnan(X)) = 0;