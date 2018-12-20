function tree = growCCT(XTrain,YTrain,bReg,options,iFeatureNum,depth)
%growTree grows a CCT, recursively calling itself until leaves are reached
%
% function CCT = growCCT(XTrain,YTrain,bReg,options,iFeatureNum,depth)
%
% Function applies greedy splitting according to the CCT algorithm and the
% provided options structure.  This is equivalent to alogrithm 2 in the
% paper if the options structure is default.  Algorithm either returns a
% leaf or forms an internal splitting node in which case the function
% recursively calls itself for each of the children, eventually returning
% the corresponding subtree.
%
% Inputs:
%    XTrain      = Array giving training features.  Data should be
%                  processed using processInputData before being passed to
%                  CCT
%    YTrain      = Output data after formatting carried out by genCCF 
%    bReg        = Whether to perform regression instead of classification.
%                  Default = false (i.e. classification).
%    options     = Options class of type optionsClassCCF.  Some fields are
%                  updated during recursion
%    iFeatureNum = Grouping of features as per processInputData.  During
%                  recursion if a feature is found to be identical across
%                  data points, the corresponding values in iFeatureNum are
%                  replaced with NaNs.
%    depth       = Current tree depth (zero based)
%
% Outputs
%   tree         = Structure containing learnt tree.  Prediction can be
%                  made using predictFromCCT
%
% Tom Rainforth 23/07/17

% Set any missing required variables

if isempty(options.mseTotal)
    options.mseTotal = var(YTrain,[],1);
end

%% First do checks for whether we should immediately terminate

N = size(XTrain,1);
% Return if one training point, pure node or if options for returning
% fulfilled.  A little case to deal with a binary YTrain is required.
bStop = (N<(max([2,options.minPointsForSplit,2*options.minPointsLeaf]))) || ...
            (isnumeric(options.maxDepthSplit) && depth>options.maxDepthSplit);

if depth>490 && strcmpi(options.maxDepthSplit,'stack')
    bStop = true;
    warning('Reached maximum depth imposed by stack limitations');
end

if bStop
    tree = setupLeaf(YTrain,bReg,options);
    return
elseif ~bReg
    % Check class variation
    sumY = sum(YTrain,1);
    bYVaries = (sumY~=0) & (sumY~=N);
    if ~any(bYVaries)
        tree = setupLeaf(YTrain,bReg,options);
        return
    end
else
    % Check if variance in Y is less than the cut off amount
    varY = var(YTrain);
    if all(varY<(options.mseTotal*options.mseErrorTolerance))
        tree = setupLeaf(YTrain,bReg,options);
        return
    end
end

%% Subsample features as required for hyperplane sampling

iCanBeSelected = fastUnique(iFeatureNum);
iCanBeSelected(isnan(iCanBeSelected))=[];
lambda = min(numel(iCanBeSelected),options.lambda);
indFeatIn = randperm(numel(iCanBeSelected),lambda);
iFeatIn = iCanBeSelected(indFeatIn);

bInMat = bsxfun(@eq,iFeatureNum(:)',sort(iFeatIn(:)));

iIn = find(any(bInMat,1));

% Check for variation along selected dimensions and resample features that
% have no variation
bXVaries = queryIfColumnsVary(XTrain(:,iIn),options.XVariationTol);

if ~all(bXVaries)
    iInNew = iIn;
    nSelected = 0;
    iIn = iIn(bXVaries);
    while ~all(bXVaries) && lambda>0
        iFeatureNum(iInNew(~bXVaries)) = NaN;
        bInMat(:,iInNew(~bXVaries)) = false;
        bRemainsSelected = any(bInMat,2);
        nSelected = nSelected+sum(bRemainsSelected);
        iCanBeSelected(indFeatIn) = [];
        lambda = min(numel(iCanBeSelected),options.lambda-nSelected);
        if lambda<1
            break
        end
        indFeatIn = randperm(numel(iCanBeSelected),lambda);
        iFeatIn = iCanBeSelected(indFeatIn);
        bInMat = bsxfun(@eq,iFeatureNum(:)',iFeatIn(:));
        iInNew = find(any(bInMat,1));
        bXVaries = queryIfColumnsVary(XTrain(:,iInNew),options.XVariationTol);
        iIn = sort([iIn,iInNew(bXVaries)]);
    end
end

if isempty(iIn)
    % This means that there was no variation along any feature, therefore
    % exit.
    tree = setupLeaf(YTrain,bReg,options);
    return
end


%% Projection bootstrap if required

if options.bProjBoot
    iTrainThis = randi(N,N,1);
    XTrainBag = XTrain(iTrainThis,iIn);
    YTrainBag = YTrain(iTrainThis,:);
else
    XTrainBag = XTrain(:,iIn);
    YTrainBag = YTrain;
end

bXBagVaries = queryIfColumnsVary(XTrainBag,options.XVariationTol);
if ~any(bXBagVaries) || ...
        (~bReg && size(YTrainBag,2)>1 && (sum(abs(sum(YTrainBag,1))>1e-12)<2)) || ...
        (~bReg && size(YTrainBag,2)==1 && any(sum(YTrainBag)==[0,size(YTrainBag,1)])) || ...
        (bReg && all(var(YTrainBag)<(options.mseTotal*options.mseErrorTolerance)))
    if ~options.bContinueProjBootDegenerate
        tree = setupLeaf(YTrain,bReg,options);
        return
    else
        XTrainBag = XTrain(:,iIn);
        YTrainBag = YTrain;
    end
end


%% Check for only having two points

if ~isempty(options.projections) && ((size(XTrainBag,1)==2) || queryIfOnlyTwoUniqueRows(XTrainBag))
    % If there are only two points setup a maximum marginal split between the points
    
    [bSplit,projMat,partitionPoint] = twoPointMaxMarginSplit(XTrainBag,YTrainBag,options.XVariationTol);
    if ~bSplit
        tree = setupLeaf(YTrain,bReg,options);
        return
    else
        bLessThanTrain = (XTrain(:,iIn)*projMat)<=partitionPoint;
        iDir = 1;
    end
else
    % Generate the new features as required
    
    if options.bRCCA
        [wZ,bZ] = genFeatureExpansionParameters(XTrainBag,options.rccaNFeatures,options.rccaLengthScale);
        fExp = makeExpansionFunc(wZ,bZ,options.rccaIncludeOriginal);
        XTrainBag = fExp(XTrainBag);
        projMat = regCCA_alt(XTrainBag,YTrainBag,options.rccaRegLambda,options.rccaRegLambda,1e-8);
        if isempty(projMat)
            projMat = ones(size(XTrainBag,2),1);
        end
        UTrain = fExp(XTrain(:,iIn))*projMat;
    else
        [projMat,yprojMat] = componentAnalysis(XTrainBag,YTrainBag,options.projections,options.epsilonCCA);
        UTrain = XTrain(:,iIn)*projMat;
    end
    
    
    %% Choose the features to use
    
    % This step catches splits based on no significant variation
    bUTrainVaries = queryIfColumnsVary(UTrain,options.XVariationTol);
    
    if ~any(bUTrainVaries)
        tree = setupLeaf(YTrain,bReg,options);
        return
    end
    
    UTrain = UTrain(:,bUTrainVaries);
    projMat = projMat(:,bUTrainVaries);
        
    if options.bUseOutputComponentsMSE && bReg && size(YTrain,2)>1 && ~isempty(yprojMat) && strcmpi(options.splitCriterion,'mse')
        VTrain = YTrain*yprojMat;
    end
    
    
    %% Search over splits using provided method
    
    nProjDirs = size(UTrain,2);
    splitGains = NaN(nProjDirs,1);
    iSplits = NaN(nProjDirs,1);
    
    for nVarAtt = 1:nProjDirs
        
        % Calculate the probabilities of being at each class in each of child
        % nodes based on proportion of training data for each of possible
        % splits using current projection
        [UTrainSort,iUTrainSort] = sort(UTrain(:,nVarAtt));
        bUniquePoints = [diff(UTrainSort,[],1)>options.XVariationTol;false];
                
        if options.bUseOutputComponentsMSE && bReg && size(YTrain,2)>1 && ~isempty(yprojMat) && strcmpi(options.splitCriterion,'mse')
            VTrainSort = VTrain(iUTrainSort,:);
        else
            VTrainSort = YTrain(iUTrainSort,:);
        end
        
        leftCum = cumsum(VTrainSort,1);
        if (size(YTrain,2)==1 || options.bSepPred ) && ~bReg
            % Convert to [class_doesnt_exist,class_exists]
            leftCum = [bsxfun(@minus,(1:N)',leftCum),leftCum]; %#ok<AGROW>
        end
        rightCum = bsxfun(@minus,leftCum(end,:),leftCum);
              
        % Calculate the metric values of the current node and two child nodes
        if ~bReg
           pL = bsxfun(@rdivide,leftCum,(1:N)');
           pR = bsxfun(@rdivide,rightCum,((N-1):-1:0)');
           
           switch options.splitCriterion
               case 'gini'
                   % Can ignore the 1 as this cancels in the gain
                   lTerm = -pL.^2;
                   rTerm = -pR.^2;
               case 'info'
                   lTerm = -pL.*log2(pL);
                   lTerm(pL==0) = 0;
                   rTerm = -pR.*log2(pR);
                   rTerm(pR==0) = 0;
               otherwise
                   error('Invalid split criterion')
           end
                
           if size(YTrain,2)==1 || options.bSepPred 
               % Add grouped terms back together
               lTerm = lTerm(:,1:end/2)+lTerm(:,end/2+1:end);
               rTerm = rTerm(:,1:end/2)+rTerm(:,end/2+1:end);
           end
           
           if ~isnumeric(options.taskWeights) && ~strcmpi(options.multiTaskGainCombination,'max')
               % No need to do anything fancy in the metric calculation
               metricLeft = sum(lTerm,2);
               metricRight = sum(rTerm,2);
           else
               % Need to do grouped sums for each of the outputs as will be
               % doing more than a simple averaging of there values
               metricLeft = cumsum(lTerm,2);
               metricLeft = metricLeft(:,[options.task_ids(2:end)-1,end])-...
                             [zeros(size(metricLeft,1),1),metricLeft(:,options.task_ids(2:end)-1)];
               metricRight = cumsum(rTerm,2);
               metricRight = metricRight(:,[options.task_ids(2:end)-1,end])-...
                   [zeros(size(metricRight,1),1),metricRight(:,options.task_ids(2:end)-1)];
           end
        else
            if strcmpi(options.splitCriterion,'mse')
                cumSqLeft = cumsum(VTrainSort.^2);
                varData = (cumSqLeft(end,:)/N)-(leftCum(end,:)/N).^2;
                if all(varData<(options.mseTotal*options.mseErrorTolerance))
                    % Total variation is less then the allowed tolerance so
                    % terminate and construct a leaf
                    tree = setupLeaf(YTrain,bReg,options);
                    return
                end
                metricLeft = calc_mse([zeros(1,size(VTrainSort,2));leftCum],cumSqLeft,VTrainSort);
                % For calculating the right need to go in additive order again
                % so go from other end and then flip
                metricRight = [zeros(1,size(VTrainSort,2));...
                    calc_mse(rightCum(end:-1:1,:),...
                    bsxfun(@minus,cumSqLeft(end,:),cumSqLeft(end-1:-1:1,:)),...
                    VTrainSort(end:-1:2,:))];
                metricRight = metricRight(end:-1:1,:);
                % No need to do the grouping for regression as each must be
                % a seperate output anyway.
            else
                error('Invalid split criterion');
            end
        end
        
        metricCurrent = metricLeft(end,:); 
        metricLeft(~bUniquePoints,:) = inf;
        metricRight(~bUniquePoints,:) = inf;
        
        % Calculate gain in metric for each of possible splits based on current
        % metric value minus metric value of child weighted by number of terms
        % in each child
        metricGain = bsxfun(@minus,metricCurrent,...
            (bsxfun(@times,(1:N)',metricLeft)...
            +bsxfun(@times,(N-1:-1:0)',metricRight))/N);
        
        % Combine gains if there are mulitple outputs.  Note that for gini,
        % info and mse, the joint gain is equal to the mean gain, hence
        % taking the mean here rather than explicitly calculating joints
        % before.
        if size(metricGain,2)>1
            if isnumeric(options.taskWeights)
                % If weights provided, weight task appropriately in terms
                % of importance.
                metricGain = bsxfun(@times,metricGain,options.taskWeights(:)');
            end
            switch options.multiTaskGainCombination
                case 'mean'
                    metricGain = mean(metricGain,2);
                case 'max'
                    metricGain = max(metricGain,[],2);
                otherwise
                    error('Invalid option for options.multiTaskGainCombination')
            end
        end
        
        % Disallow splits that violate the minimum number of leaf points
        metricGain(1:(options.minPointsLeaf-1)) = -inf;
        metricGain((end-(options.minPointsLeaf-1)):end) = -inf; % Note that end is never chosen anyway
        
        % Randomly sample from equally best splits
        [splitGains(nVarAtt),iSplits(nVarAtt)] = max(metricGain(1:end-1));
        iEqualMax = find(abs(metricGain(1:end-1)-splitGains(nVarAtt))<(10*eps));
        if isempty(iEqualMax)
            iEqualMax = 1;
        end
        iSplits(nVarAtt) = iEqualMax(randi(numel(iEqualMax)));
        
    end
    
    % If no split gives a positive gain then stop
    if max(splitGains)<0
        tree = setupLeaf(YTrain,bReg,options);
        return
    end
    
    % Establish between projection direction
    maxGain = max(splitGains);
    iEqualMax = find(abs(splitGains-maxGain)<(10*eps));
    % Use given method to break ties
    if strcmpi(options.dirIfEqual,'rand')
        iDir = iEqualMax(randi(numel(iEqualMax)));
    elseif strcmpi(options.dirIfEqual,'first')
        iDir = iEqualMax(1);
    else
        error('invalid dirIfEqual');
    end
    iSplit = iSplits(iDir);
    
    
    %% Establish partition point and assign to child
    
    UTrain = UTrain(:,iDir);
    UTrainSort = sort(UTrain);
    
    % The convoluted nature of the below is to avoid numerical errors
    uTrainSortLeftPart = UTrainSort(iSplit);
    UTrainSort = UTrainSort-uTrainSortLeftPart;
    partitionPoint = UTrainSort(iSplit)*0.5+UTrainSort(iSplit+1)*0.5;
    partitionPoint = partitionPoint+uTrainSortLeftPart;
    UTrainSort = UTrainSort+uTrainSortLeftPart; %#ok<NASGU>
    
    bLessThanTrain = UTrain<=partitionPoint;
    
    if ~any(bLessThanTrain) || all(bLessThanTrain)
        error('Suggested split with empty');
    end
    
end

%% Recur tree growth to child nodes and constructs tree struct
% to return
tree.bLeaf = false;
tree.Npoints = N;
tree.mean = mean(YTrain,1);

if bReg
    if ~isempty(options.org_stdY)
        tree.mean = tree.mean.*options.org_stdY;
    end
    if ~isempty(options.org_muY)
        tree.mean = tree.mean+options.org_muY;
    end
end

treeLeft = growCCT(XTrain(bLessThanTrain,:),YTrain(bLessThanTrain,:),bReg,options,iFeatureNum,depth+1);
treeRight = growCCT(XTrain(~bLessThanTrain,:),YTrain(~bLessThanTrain,:),bReg,options,iFeatureNum,depth+1);
tree.iIn = iIn;
if options.bRCCA && exist('fExp','var')
    tree.featureExpansion = fExp;
end
tree.decisionProjection = projMat(:,iDir);
tree.paritionPoint = partitionPoint;
tree.lessthanChild = treeLeft;
tree.greaterthanChild = treeRight;

end

function tree = setupLeaf(YTrain,bReg,options)
% Update tree struct to make node a leaf

tree.bLeaf = true;
tree.Npoints = size(YTrain,1);
tree.mean = mean(YTrain,1);

if bReg
    % TODO add other possible leaf models
    
    tree.std_dev = std(YTrain,[],1);
    
    % If a mapping has been applied, invert it
    if ~isempty(options.org_stdY)
        tree.mean = tree.mean.*options.org_stdY;
        tree.std_dev = tree.std_dev.*options.org_stdY;
    end
    if ~isempty(options.org_muY)
        tree.mean = tree.mean+options.org_muY;
    end
end



end

function f = makeExpansionFunc(wZ,bZ,bIncOrig)
    if bIncOrig
        f = @(x) [x,random_feature_expansion(x,wZ,bZ)];
    else
        f = @(x) random_feature_expansion(x,wZ,bZ);
    end
end

function value = calc_mse(cumtotal,cumsq,YTrainSort)
% CMSE(j)=sigma(j)/j-(1/j^2)(mu(j-1)^2+Y(j)^2+2mu(j-1)Y(j))
%   where sigma(j) = sum_{i=1}^j Y(j)^2 i.e. cumsq(j)
%   and      mu(j) = sum_{i=1}^j Y(j) i.e. cumtotal(j+1)
value = bsxfun(@rdivide,cumsq,(1:size(YTrainSort,1))')-...
        bsxfun(@rdivide, cumtotal(1:end-1,:).^2+YTrainSort.^2+2*cumtotal(1:end-1,:).*YTrainSort...
                       , ((1:size(YTrainSort,1)).^2)');                                            
end