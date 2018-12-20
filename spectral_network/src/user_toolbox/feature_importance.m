function [importance,errors,errors_unperm] = feature_importance(CCF, X, Y, bOutOfBag)
% Estimate for the relative feature importances.
%
% Can use with a single trained forest when using CCF-Bag for which we 
% exploit the out of bag estimates.  Alternatively can do a cross-
% validation where the variable importance is averaged over doing this
% calculation provided X and Y as one fold and the CCF is trained on the
% other folds.
%
% Inputs: CCF = Trained forest
%         X = Input data to test accuracy on.  If using a single CCF-Bag
%             then set X to XTrain.  If doing a cross validation then set X
%             to XTest and average this metric across the folds.
%         Y = Corresponding output data
%         bOutOfBag = Whether to use the out of bag predictions to
%               calculate the importance.  If true must be using CCF-Bag.
%               Default is true when exists and false otherwise.
%
% Outputs: importance = estimated variable importance.  Corresponds to
%               (E_d-E_b)/E_b where E_b is the predictive error using the
%               original X and X_d is the error when dimension d of X is
%               randomly permuted.  See Section 10 of Random Forests
%               (Breiman 2001).
%          errors = Raw errors for each feature permutation (i.e. E_ds)
%          errors_unperm = Error without permutation (i.e. E_b)
%           
% Example calls:
% % 1) Using single CCF-Bag and out of bag estimates
%   [X,Y] = loadProvidedDataset('vowelc');
%   CCF = genCCF(500,X,Y,false,optionsClassCCF.defaultOptionsCCFBag);
%   [importance,errors,errors_unperm] = feature_importance(CCF,X,Y);   
%
% % 2) Cross validation with CCF
%   [X,Y] = loadProvidedDataset('iris');
%   nFolds = 10;
%   [iTrain, iTest] = setupCrossValSampleIds(size(X,1),nFolds);
%   errors = NaN(size(X,2),nFolds);
%   errors_unperm = NaN(1,nFolds);
%   % Calculate the error for each fold
%   for n=1:nFolds;
%       CCF = genCCF(500,X(iTrain{n},:),Y(iTrain{n},:));
%       [~,errors(:,n),errors_unperm(n)] = feature_importance(CCF,X(iTest{n},:),Y(iTest{n},:));
%   end
%   % Take the importance as the average error incease divided by the
%   % average error
%   importance = mean(bsxfun(@minus,errors,errors_unperm),2)/mean(errors_unperm);
%
% Tom Rainforth 29/06/17

if ~exist('bOutOfBag','var')
    bOutOfBag = isfield(CCF,'outOfBagError') && isnumeric(CCF.outOfBagError);
end

X = replicateInputProcess(X,CCF.inputProcessDetails);

nTrees = numel(CCF.Trees);
bReg = CCF.bReg;
YpermPreds = cell(nTrees,1);

for n=1:nTrees
    YpermPreds{n} = randperm_preds(CCF.Trees{n}, X, bOutOfBag);
end

[N,D] = size(X);

errors = cell(D,1);

for d=1:D+1
    cumEst = zeros(N,size(YpermPreds{1}{d},2));
    nEsts = zeros(N,1);
    if bOutOfBag
        for nTO = 1:nTrees
            cumEst(CCF.Trees{nTO}.iOutOfBag,:) = cumEst(CCF.Trees{nTO}.iOutOfBag,:)+YpermPreds{nTO}{d};
            nEsts(CCF.Trees{nTO}.iOutOfBag) = nEsts(CCF.Trees{nTO}.iOutOfBag)+1;
        end
        overall_est = bsxfun(@rdivide,cumEst,nEsts);
    else
        for nTO = 1:nTrees
            cumEst = cumEst+YpermPreds{nTO}{d};
        end
        overall_est = bsxfun(@rdivide,cumEst,nTrees);
    end
    if bReg
        errors{d} = mean(nanmean((overall_est-Y).^2,1));
    elseif CCF.options.bSepPred
        errors{d} = mean((1-nanmean((overall_est>0.5)==Y,1)));
    else    
        Y_preds = treeOutputsToForestPredicts(CCF,reshape(overall_est,size(overall_est,1),1,[]));      
        if max(Y(:))==1 && size(Y,2)~=size(Y_preds,2)
            Y = sum(bsxfun(@times,Y,1:size(Y,2)),2);
        end
        errors{d} = mean(1-nanmean(Y_preds==Y,1));
    end
end

errors_unperm = errors{end};
errors = errors(1:end-1);
errors = [errors{:}];
if size(errors,1) == 1
    errors = errors';
end

errors = mean(errors,2);
errors_unperm = mean(errors_unperm,2);
importance = bsxfun(@rdivide,bsxfun(@minus,errors,errors_unperm),errors_unperm);
importance = mean(importance,2);
importance(isnan(importance)) = 0;
