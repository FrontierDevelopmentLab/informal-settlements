function [A,B,r] = regCCA_alt(X,Y,gammaX,gammaY,corrTol)
%regCCA Fast regularized CCA.  Used when doing kernel cca 

D = size(X,2);
K = size(Y,2);

C = cov([X,Y]);
Cxx = C(1:D,1:D)+gammaX*eye(D);
Cyy = C((D+1):end,(D+1):end)+gammaY*eye(K);
Cxy = C(1:D,(D+1):end);

Cxx = 0.5*(Cxx+Cxx');
Cyy = 0.5*(Cyy+Cyy');

invCholCxx = chol(Cxx)\eye(D);
invCholCyy = chol(Cyy)\eye(K);

T = invCholCxx'*Cxy*invCholCyy;

if D>=K
    [L,S,D] = svd(T,0);
    r = diag(S);
    A = invCholCxx*L;
    B = invCholCyy*D;
else
    [L,S,D] = svd(T',0);
    r = diag(S);
    A = invCholCxx*D;
    B = invCholCyy*L;
end

bGreaterThanTol = abs(r)>abs(corrTol*max(abs(r)));
A = A(:,bGreaterThanTol);
B = B(:,bGreaterThanTol);
r = r(bGreaterThanTol);