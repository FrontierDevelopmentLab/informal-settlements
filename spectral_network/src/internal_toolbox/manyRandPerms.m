function terms = manyRandPerms(nVarTot,nVarSel,nTimes)
% Generate a number of random permutations

terms = NaN(nTimes,nVarSel);
for n=1:nTimes
    terms(n,:) = randperm(nVarTot,nVarSel);
end
end