function uX = fastUnique(X)
% Lite version of matlabs unique that is faster

uX = sort(X);
if isrow(X)
    uX = uX([true,diff(uX)~=0]);
else
    uX = uX([true;diff(uX)~=0]);
end