function [errorCCF, errorRF,yPredsCCF,yPredsRF,iTrain,iTest] = ...
    crossValTests(X,Y,bReg,bOrdinal,nFolds,nTrees,optionsFor,bPrint,bDoRF,iTrain,iTest)
%crossValTests Performs crossvalidation tests using CCFs
%
% errorCCF = crossValTests(X,Y,bReg)
%
% Required inputs:
%       X = Feature array
%       Y = Class labels
%       bReg = Whether regression (including multivariate regression)
%              problem or classification
%
%
% Optional Inputs:
%
% [errorCCF, errorRF,yPredsCCF,yPredsRF,iTrain,iTest] = ...
%    crossValTests(X,Y,bOrdinal,nFolds,nTrees,optionsFor,bPrint,bDoRF,...
%                                                         iTrain,iTest)
%
%   bOrdinal = row vector of logicals indicating whether the feature is
%              ordinal (true) or an unordered categorical requiring
%              expansion (false).  Default is all true
%     nFolds = Number of folds to carry out.  Default = 10
%     nTrees = Number of trees.  Default = 500
% optionsFor = An options object from optionsClassCCF.m
%     bPrint = Prints out accuracies if true. Default = true
%      bDoRF = Toggles whether to also carry out the RF calculation.
%              Default = true;
%     iTrain = Cell array of training indices for each fold.  If not
%              provided then these are sampled using a standard cross
%              validation
%      iTest = As iTrain but corresponding to test indices
%
% Outputs:
%    errorCCF = Errors for CCF.  Rows are different folds, columns are
%               different outputs.
%    errorRF = Errors for RF if requested.
%   yPredsCCF,yPredsRF = corresponding raw predicts, cell array where each
%              cell is a different fold
%   iTrain,iTest = Generated train / test indices
%
% Tom Rainforth 24/07/17

if ~exist('bOrdinal','var') || isempty(bOrdinal)
    bOrdinal = true(1,size(X,2));
end

if ~exist('nFolds','var') || isempty(nFolds)
    nFolds = 10;
end

if ~exist('nTrees','var') || isempty(nTrees)
    nTrees = 500;
end

if ~exist('optionsFor','var')
    optionsFor = [];
end

if ~exist('bPrint','var') || isempty(bPrint)
    bPrint = true;
end

if ~exist('bDoRF','var') || isempty(bDoRF)
    bDoRF = true;
end

if ~exist('iTrain','var') || ~exist('iTest','var') || isempty(iTrain) || isempty(iTest)
    [iTrain, iTest] = setupCrossValSampleIds(size(X,1),nFolds);
end

errorCCF = cell(nFolds,1);
errorRF = cell(nFolds,1);

yPredsCCF = cell(nFolds,1);
yPredsRF = cell(nFolds,1);
% 
% if ~bReg
%     if isempty(optionsFor)
%         Y = classExpansion(Y,size(X,1),optionsClassCCF.defaultOptions);
%     else
%         Y = classExpansion(Y,size(X,1),optionsFor);
%     end
% end

if isempty(optionsFor)
    if bReg
        optionsRF = optionsClassCCF.defaultOptionsRFReg;
    else
        optionsRF = optionsClassCCF.defaultOptionsRF;
    end
else
    optionsRF = optionsFor;
    optionsRF.bBagTrees = true;
    optionsRF.bProjBoot = false;
    optionsRF.projections = struct('Original',true);
    optionsRF.treeRotation = 'none';
    if ischar(optionsRF.lambda) && strcmpi(optionsRF.lambda,'all')
        optionsRF.lambda = 'log';
    end
end

scaling = var(Y);

for n=1:nFolds
    tC = tic;
    CCF = genCCF(nTrees,X(iTrain{n},:),Y(iTrain{n},:),bReg,optionsFor,[],[],[],bOrdinal);
    timeCCFtrain(n) = toc(tC);
    tC = tic;
    yPreds = predictFromCCF(CCF,X(iTest{n},:));
    timeCCFtest(n) = toc(tC);
    if ~iscell(Y) && size(Y,2)~=size(yPreds,2) && max(Y(:))==1
        Y = sum(bsxfun(@times,Y,1:size(Y,2)),2);
    end
    yPredsCCF{n} = yPreds;
    if ~bReg
        if iscell(Y)
            % FIXME multiple output case not supported here
            errorCCF{n} = 100*(1-mean(cellfun(@strcmpi,CCF.classNames(yPreds),Y(iTest{n}))))';
        else
            errorCCF{n} = 100*(1-mean(yPreds==Y(iTest{n},:)))';
        end
        dispMessage = ['Fold ' num2str(n) ', test misclassifcation rate (lower is better): CCF = ' ...
            num2str(errorCCF{n}',4) '%'];
    else
        errorCCF{n} = 100*mean(((yPreds-Y(iTest{n},:)).^2)./scaling,1)';
        dispMessage = ['Fold ' num2str(n) ', mse scaled by std dev of Y (lower is better): CCF  = ' ...
            num2str(errorCCF{n}',4) '%'];
    end
    if bDoRF
        tC = tic;
        CCF = genCCF(nTrees,X(iTrain{n},:),Y(iTrain{n},:),bReg,optionsRF,[],[],[],bOrdinal);
        timeRFtrain(n) = toc(tC);
        tC = tic;
        yPreds = predictFromCCF(CCF,X(iTest{n},:));
        timeRFtest(n) = toc(tC);
        yPredsRF{n} = yPreds;
        if ~bReg
            if iscell(Y)
                % FIXME multiple output case not supported here
                errorRF{n} = 100*(1-mean(cellfun(@strcmpi,CCF.classNames(yPreds),Y(iTest{n}))))';
            else
                errorRF{n} = 100*(1-mean(yPreds==Y(iTest{n},:)))';
            end
            dispMessage = [dispMessage, ', RF = ' ...
                num2str(errorRF{n}',4) '%'];
        else
            errorRF{n} = 100*mean(((yPreds-Y(iTest{n},:)).^2)./scaling,1)';
            dispMessage = [dispMessage, ', RF  = ' ...
                num2str(errorRF{n}',4) '%'];
        end
        %         timeMessage = ['Train time: CCF  = ' num2str(timeCCFtrain,4) 's, RF = ' num2str(timeRFtrain,4) 's\n'...
        %             'Test time: CCF  = ' num2str(timeCCFtest,4) 's, RF = ' num2str(timeRFtest,4) 's'];
    elseif bPrint
        timeMessage = ['CCF train time = ' num2str(timeCCFtrain,4) 's, CCF test time = ' ...
            num2str(timeCCFtest,4) 's'];
    end
    
    if bPrint
        disp(dispMessage);
        %disp(sprintf(timeMessage));
    end
end

errorCCF = [errorCCF{:}]';
if bDoRF
    errorRF = [errorRF{:}]';
end

if bPrint
    disp(sprintf(['Average CCF train time = ' num2str(mean(timeCCFtrain),4) 's, test time = ' num2str(mean(timeCCFtest),4) 's\n'...
        'Average RF train time = ' num2str(mean(timeRFtrain),4) 's, test time = ' num2str(mean(timeRFtest),4) 's'])); %#ok<DSPS>
    if bReg
        common_message = 'Scaled mean squared error';
    else
        common_message = 'Missclassfication rate';
    end
    disp([common_message ' CCF = ' num2str(mean(errorCCF)) ' +/- ' num2str(std(errorCCF))]);
    if bDoRF
        disp([common_message ' RF = ' num2str(mean(errorRF)) ' +/- ' num2str(std(errorRF))]);
    end
    if size(errorCCF,2)~=1
        disp(['Output average ' common_message ' CCF = ' num2str(mean(mean(errorCCF))) ' +/- ' num2str(mean(std(errorCCF,[],1)))]);
        if bDoRF
            disp(['Output average ' common_message ' RF = ' num2str(mean(mean(errorRF))) ' +/- ' num2str(mean(std(errorRF,[],1)))]);
        end
    end
    disp(' ');
end
end