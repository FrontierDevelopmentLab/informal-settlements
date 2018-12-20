function plotClasses(X,Y,bNewPlot)
% Plots data into seperate classes.
%
% Inputs:
%   X = matrix of features (only plots first two dimensions)
%   Y = matrix of logicals or vector of class indices
%   bNewPlot = Where to add to currently active plot or make a new plots
%              (default = true)

if ~exist('bNewPlot','var')
    bNewPlot = true;
end

colors = {'b','g','r','k','c','y','m'};
markers = {'x','+','o','*','square','diamond','pentagram','hexagram'};

if size(Y,2)~=1
    Y = sum(bsxfun(@times,Y,1:size(Y,2)),2);
elseif islogical(Y)
    Y = Y+1;
end

if bNewPlot
    figure
end
hold on

for k=1:max(Y)
    plot(X(Y==k,1),X(Y==k,2),[colors{mod(k-1,numel(colors))+1} markers{mod(k,numel(markers))+1}]);
end
