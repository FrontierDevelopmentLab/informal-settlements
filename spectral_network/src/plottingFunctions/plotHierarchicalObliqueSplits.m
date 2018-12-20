function plotHierarchicalObliqueSplits(CCT,inputProcessDetails,x1Lims,x2Lims,bPlotEdges,colors,alpha)
% Creates a plot of the hierarchical splits for a single CCT, such as used
% in the paper to create the plot for the spirals CCT.
%
% Inputs:
%   CCT = Single tree to plot splits for.
%   inputProcessDetails = If genSingleCCT was used to make the CCT this can
%          be left empty as the details will be stored in the CCT.
%          Otherwise it is a field of the forest the CCT was taken from.
%   xLims, yLims = plot limits for x and y, dictates area that will be
%           filled in.
%   bPlotEdges = Whether to include black lines for the edges (true) or 
%       just fill in the space (false). Default = true
%   colors = Colors for each class, default is provided.
%   alpha = face alpha for the fill plots.  Default = 0.1

if ~exist('inputProcessDetails','var') || isempty(inputProcessDetails)
    inputProcessDetails = CCT.inputProcessDetails;
end

[starts,ends,labels] = CTT_TreeToLines(CCT,inputProcessDetails);

figure

if ~exist('bPlotEdges','var') || isempty(bPlotEdges)
    bPlotEdges = true;
end

if ~exist('alpha','var') || isempty(alpha)
    alpha = 0.1;
end

if ~exist('colors','var') || isempty(colors)
    colors = {'b','r','k','c','g','w','b','r'};
end

x1bord = [x1Lims(1);x1Lims(2);x1Lims(2);x1Lims(1);x1Lims(1)];
x2bord = [x2Lims(1);x2Lims(1);x2Lims(2);x2Lims(2);x2Lims(1)];

hold on;
plot(x1bord,x2bord,'k');
axis([x1Lims(1)-0.05,x1Lims(2)+0.05,x2Lims(1)-0.05,x2Lims(2)+0.05])

plotSplitRec(starts,ends,labels,x1bord,x2bord,colors,bPlotEdges,alpha);

end

function plotSplitRec(sp,ep,la,polyx1,polyx2,colors,bPlotEdges,alpha)
if isempty(sp)
    if bPlotEdges
        fill(polyx1,polyx2,colors{la},'FaceAlpha',alpha)
    else
        fill(polyx1,polyx2,colors{la},'FaceAlpha',alpha,'EdgeColor','none')
    end
    return
end

[x1i,x2i,ii] = polyxpoly([sp{1}(1),ep{1}(1)],[sp{1}(2),ep{1}(2)],polyx1,polyx2);

polyleftx1 = [polyx1(1:ii(1,2)); x1i(1); x1i(2); polyx1((ii(2,2)+1):end)];
polyleftx2 = [polyx2(1:ii(1,2)); x2i(1); x2i(2); polyx2((ii(2,2)+1):end)];

polyrightx1 = [x1i(1); polyx1((ii(1,2)+1):(ii(2,2))); x1i(2); x1i(1)];
polyrightx2 = [x2i(1); polyx2((ii(1,2)+1):(ii(2,2))); x2i(2); x2i(1)];

if abs(ep{1}(2)-sp{1}(2))<1e-10
    projVec = [0,1];
else
    projVec = [1, -(ep{1}(1)-sp{1}(1))/(ep{1}(2)-sp{1}(2))];
    projVec = projVec*sign(projVec(2));
end

maxAwayLeft = max([polyleftx1,polyleftx2]*projVec');
maxAwayRight = max([polyrightx1,polyrightx2]*projVec');

if maxAwayLeft<maxAwayRight
    plotSplitRec(sp{2},ep{2},la{2},polyleftx1,polyleftx2,colors,bPlotEdges,alpha)
    plotSplitRec(sp{3},ep{3},la{3},polyrightx1,polyrightx2,colors,bPlotEdges,alpha)
else
    plotSplitRec(sp{3},ep{3},la{3},polyleftx1,polyleftx2,colors,bPlotEdges,alpha)
    plotSplitRec(sp{2},ep{2},la{2},polyrightx1,polyrightx2,colors,bPlotEdges,alpha)
end
end


function [starts,ends,labels] = CTT_TreeToLines(CCT,inputProcessDetails)

if CCT.bLeaf
    starts = {};
    ends = {};
    [~,labels] = max(CCT.mean);
    return
end

grad = -CCT.decisionProjection(1)/CCT.decisionProjection(2);

if ~isfinite(grad) || (abs(grad)>1e12)
    startPoint = [CCT.paritionPoint, -1e6];
    endPoint = [CCT.paritionPoint, 1e6];
else
    int = CCT.paritionPoint/CCT.decisionProjection(2);
    startPoint = [-1e6,grad*(-1e6)+int];
    endPoint = [1e6,grad*1e6+int];
end

startPoint = zScoreToX(startPoint,InputProcessDetails.mu_XTrain,InputProcessDetails.std_XTrain);
endPoint = zScoreToX(endPoint,inputProcessDetails);
labelT = [];

if CCT.decisionProjection(2)>0
    [startLeft, endLeft, labelsLeft] = CTT_TreeToLines(CCT.lessthanChild,inputProcessDetails);
    [startRight, endRight, labelsRight] = CTT_TreeToLines(CCT.greaterthanChild,inputProcessDetails);
else
    [startLeft, endLeft, labelsLeft] = CTT_TreeToLines(CCT.greaterthanChild,inputProcessDetails);
    [startRight, endRight, labelsRight] = CTT_TreeToLines(CCT.lessthanChild,inputProcessDetails);
end

starts = [{startPoint};{startLeft};{startRight}];
ends = [{endPoint};{endLeft};{endRight}];
labels = [{labelT};{labelsLeft};{labelsRight}];

end