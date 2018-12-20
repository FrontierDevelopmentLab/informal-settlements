function A = makeSureString(A,nSigFigTol)
% Ensure that all numerical values are strings

bNum = isnumeric(A);
A(bNum) = num2str(A(bNum),nSigFigTol);
end