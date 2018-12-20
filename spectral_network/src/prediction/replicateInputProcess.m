function X = replicateInputProcess(Xraw,InputProcessDetails)
%replicateInputProcess
%
% function X = replicateInputProcess(X,InputProcessDetails)
% 
% This can be used to create an anonymous function that applies the same
% data transformation as was done by on the training data to new data.
% InputProcessDetails is the structure output from processInputData.m and
% stored in the forest.
%
% Tom Rainforth 04/08/15

bOrdinal = InputProcessDetails.bOrdinal;
Cats = InputProcessDetails.Cats;

if size(Xraw,2)~=numel(bOrdinal)
    error('Incorrect number of features');
end

if istable(Xraw)
    try
        Xraw = table2array(Xraw);
    catch
        Xraw = table2cell(Xraw);
    end
end

X = Xraw(:,bOrdinal);
if iscell(X)
    bNumeric = cellfun(@isnumeric,X);
    Xrep = cellfun(@(x) sscanf(x,'%f'), X(~bNumeric),'UniformOutput',false);
    Xrep(cellfun(@isempty,Xrep)) = {NaN};
    X(~bNumeric) = Xrep;
    X = cell2mat(X);
end

XCat = Xraw(:,~bOrdinal);
if iscell(XCat)
    XCat = makeSureString(XCat,10);
end

for n=1:size(XCat,2)
    nCats = numel(Cats{n});
    if nCats==1
        continue
    end
    sizeSoFar = size(X,2);
    X = [X,zeros(size(X,1),nCats)]; %#ok<AGROW>
    for c=1:nCats
        if iscell(Cats{n})
            X(strcmp(XCat(:,n),Cats{n}{c}),(sizeSoFar+c)) = 1;
        else
            X(XCat(:,n)==Cats{n}(c),(sizeSoFar+c)) = 1;
        end
    end
end

X = bsxfun(@rdivide,bsxfun(@minus,X,InputProcessDetails.mu_XTrain),InputProcessDetails.std_XTrain);

if InputProcessDetails.bNaNtoMean
    X(isnan(X)) = 0;
end

end