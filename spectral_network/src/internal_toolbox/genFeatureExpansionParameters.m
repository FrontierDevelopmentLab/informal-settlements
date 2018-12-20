function [w,b] = genFeatureExpansionParameters(X,nF,s)
% generates random feature parameters for kernel CCA

if nargin<3
    s = 0.1;
end

M = size(X,2);
w = s*randn(M,nF);
b = 2*pi*rand(1,nF);

end