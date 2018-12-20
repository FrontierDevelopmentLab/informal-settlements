function X = random_missing_vals(X,mu,sig)
% Randomly assigns missing values to draws from a normal with the mean and
% standard deviation of the data.

if ~exist('mu','var') || isempty(mu)
    mu = 0;
end

if ~exist('sig','var') || isempty(sig)
    sig = 1;
end

bNaN = isnan(X);
nRands = sum(bNaN(:));
if nRands~=0
    X(bNaN) = sig*randn(nRands,1)+mu;
end

end