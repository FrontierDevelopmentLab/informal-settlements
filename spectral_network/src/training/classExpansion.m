function [Y, classes, optionsFor] = classExpansion(Y,N,optionsFor)
%classExpansion Ensures that class data is in its binary expansion format
%
% [Y, classes, optionsFor] = classExpansion(Y,N,optionsFor)
%
% Inputs: 
%    Y       = Class information, can be a binary expansion, a numerical
%              vector of labels or a cell array of numerical or string 
%              labels.  For multiple inputs, should instead be a 1xV cell
%              array where each cell is of a type required for single
%              input.
%    N       = Number of datapoints.
%    optionsFor = Forest options
%            
% Outpus: 
%    Y       = Y in binary expansion format
%    classes = Names of classes.  In CCT only the class index is stored and
%              so this is used to convert to the original name.
%    optionsFor = Updated forest options, e.g. because bSepPred has been
%              switched on because non-mutually exclusive classes.
%
% Tom Rainforth 22/06/15

%warning('Do class expansion for non-logical multi inputs.');

if size(Y,1)==N && size(Y,2)==1
    assert(~optionsFor.bSepPred,'Seperate in-out prediction is only valid when Y is a logical array');
    [classes,~,Yindexes] = unique(Y);
    Y = false(size(Yindexes,1),numel(classes));
    for k=1:numel(classes)
        Y(:,k) = k==Yindexes;
    end
    optionsFor.task_ids = 1;
elseif iscell(Y)
    assert(size(Y,1)==1,'If Y is a cell it should either be Nx1 for a single output or 1xV for various outputs, were v=1:V is an array for that output');
    assert(~optionsFor.bSepPred,'Seperate in-out prediction is only valid when Y is a logical array');
    classes = cell(1,numel(Y));
    Ycell = cell(1,numel(Y));
    for n=1:numel(Y)
        [Ycell{n},classes{n},optionsFor] = classExpansion(Y{n},N,optionsFor);
        assert(optionsFor.task_ids ==1,'When providing cell array of outputs, each cell must contain only a single task');
        assert(~optionsFor.bSepPred,'Seperate in-out prediction is only valid when Y is a logical array');
    end
    y_sizes = cellfun(@(x) size(x,2), Ycell);
    Y = cell2mat(Ycell);
    optionsFor.task_ids = 1+[0,cumsum(y_sizes(1:end-1))];
elseif islogical(Y) || (max(Y(:))==1 && min(Y(:))==0)
    N_c_present = cumsum(Y,2); 
    if all(N_c_present(:,end)==1) && ~optionsFor.bSepPred
        optionsFor.task_ids = 1;
        classes = (1:size(Y,2))';        
    else
        if ~optionsFor.bSepPred
            optionsFor.bSepPred = true;
            warning('Providing a logical array with varying number of active classes, setting bSepPred to true.  For multi-output classification use array of class indices where each column is an output');
        end
        optionsFor.task_ids = 1:size(Y,2);
        classes = repmat({[false,true]},1,size(Y,2));
    end 
else
    assert(~optionsFor.bSepPred,'Seperate in-out prediction is only valid when Y is a logical array');
    classes = cell(1,size(Y,2));
    Ycell = cell(1,size(Y,2));
    for n=1:size(Y,2)
        [Ycell{n},classes{n},optionsFor] = classExpansion(Y(:,n),N,optionsFor);
    end
    y_sizes = cellfun(@(x) size(x,2), Ycell);
    Y = cell2mat(Ycell);
    optionsFor.task_ids = 1+[0,cumsum(y_sizes(1:end-1))];
end

if size(classes,1)>(N-2)
    error('More than n_data_points-2 classes appear to be present.  Make sure no datapoints with missing output or regression option on!');
end