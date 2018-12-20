function [iTrain, iTest] = setupCrossValSampleIds(nDataPoints,nFolds)
%setupCrossValSampleIds Returns a set of ids for cross validation
%
% [iTrain, iTest] = setupCrossValSampleIds(nDataPoints,nFolds)
%
% iTrain and iTest will each be cell arrays of size nFolds x 1 with each
% cell containing a set of ids corresponding the training and test data
% points based on a cross validation.  The folds are setup up randomly
% without attempt to create even representation of the classes.  It is
% ensured that each train / test set forms a full representation of the
% data, with the final fold potentially having repeated test points if
% nFolds is not a factor of nDataPoints.
%
% Tom Rainforth 27/07/15

fold = cell(nFolds,1);
foldSize = round(nDataPoints/nFolds);
fold{1} = randperm(nDataPoints,foldSize);
left = setdiff(1:nDataPoints,fold{1});
for nF=2:(nFolds-1)
    if numel(left)<foldSize
        extraForFold = datasample(setdiff(1:nDataPoints,left),foldSize-numel(left),2,'Replace',false);
        left = [left,extraForFold];
    end
    fold{nF} = datasample(left,foldSize,2,'Replace',false);
    left = setdiff(left,fold{nF});
end

% Make usre that the 10 fold is the same size
if numel(left)>foldSize
    left = datasample(left,foldSize,2,'Replace',false);
elseif numel(left)<foldSize
    extraForFold = datasample(setdiff(1:nDataPoints,left),foldSize-numel(left),2,'Replace',false);
    left = [left,extraForFold];
end
fold{nFolds} = left;

iTrain = cell(nFolds,1);
iTest = cell(nFolds,1);

for n=1:nFolds
    iTest{n} = fold{n};
    iTrain{n} = setdiff(1:nDataPoints,iTest{n});
end

end