function Q = randomRotation(N)
%randomRotation Random rotation matrix of given dimension.
% See "Random Rotation Ensembles" by Blaser et al 2015.
%
% First step:
%  Q * diag(sign(diag(R))) * diag(sign(diag(R))) * R = randn(N) where R is
%  upper triangular gives matrixes for which Q'=inv(Q).  The diag terms are
%  to give another QR decomposition qr=randn(N) where 
%  r = diag(sign(diag(R))) * R only has a positive diagonal. As 
%  diag(sign(diag(R)))*diag(sign(diag(R))) = I, then QR=qr=rand(N).
%
% Second step:
%  Q might have det = -1 here.  By setting the first row of Q to be its
%  original negative then and calling this Q2, then Q2R2 = Q, Q2R2R =
%  randn(N), R2R is upper triangular with positive only diagonal, and
%  det(Q2)=1, whilst retaining that Q2'=inv(Q).  Thus Q2 is randomly
%  normally distributed

[Q,R] = qr(randn(N));
Q = Q*diag(sign(diag(R))); % Q * diag(sign(diag(R))) * diag(sign(diag(R))) * R = randn(N) bu
detR = det(Q);
if round(detR)==-1
    Q(:,1) = -Q(:,1);
end
end