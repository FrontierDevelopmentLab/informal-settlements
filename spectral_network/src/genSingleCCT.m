function [CCT,varargout] = ...
    genSingleCCT(XTrain,YTrain,bReg,options,XTest,iFeatureNum,bOrdinal)
%genSingleCCT Generate a single canonical correlation tree
%
% Creates a single canonical correlation tree (CCT) which use splits based
% on a CCA analysis between the input data and labels.  Usage is as per
% genCCF but returns a single CCT instead of a full CCF.  Prediction can
% then be done using predictFromCCT.  Other outputs are as per genCCF
%
% Tom Rainforth 23/07/17

if ~exist('bReg','var')
    bReg = [];
end

if ~exist('XTest','var')
    XTest = [];
end

if ~exist('options','var')
    options = optionsClassCCF.defaultOptionsForSingleCCTUsage;
end

if ~exist('iFeatureNum','var')
    iFeatureNum = [];
end

if ~exist('bOrdinal','var')
    bOrdinal = [];
end

options.bUseParallel = false;

% Strange naming it to avoid wasting space by storing two instances
[CCT,varargout{1:(nargout-1)}] = deal(genCCF(1,XTrain,YTrain,bReg,options,XTest,true,iFeatureNum,bOrdinal));

% This is a bit weird to avoid duplicating the tree for reduced memory
other_outs = rmfield(CCT,'Trees');
CCT = CCT.Trees{1};
other_fields = fields(other_outs);
for n=1:numel(other_fields)
    CCT.(other_fields{n}) = other_outs.(other_fields{n});
end

end
