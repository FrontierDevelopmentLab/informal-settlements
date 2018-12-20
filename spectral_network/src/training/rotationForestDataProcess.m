function [R, muX, U] = rotationForestDataProcess(X,Y,M,prop_points_subsample,prop_classes_eliminate)
%Calculate a rotation forest projection matrix for processing a tree
%
% [R, muX, U] = rotationForestDataProcess(X,Y,M,prop_points_subsample,...
%                                           prop_classes_eliminate)
%
% Carries out the random pca-based projection required for training each
% tree used in a rotation forest as per
%
% Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. 
%   "Rotation forest: A new classifier ensemble method." 
% IEEE transactions on pattern analysis and machine intelligence 28.10 
% (2006): 1619-1630.
%
% Currently supports classification, regression and multiple output 
% classification or regression.  I haven't done much testing for multiple 
% output classification though so could have unexpected behaviour.
%
% Inputs:
%   X = Input features
%   Y = Processed outputs, used for random class elimination for
%       classification and multi-output classification problems.  No such
%       elimination is used for regression problems
%   M = Size of feature subsets to use for PCA, as per M in the orignal
%       paper
%   prop_points_subsample = Size of bootstrap sample taken for PCA as a
%       proportion of full dataset size.
%   prop_classes_eliminate = Proportion of classes to eliminate for each
%       PCA calculation (classes are chosen randomly).  This is mentioned
%       in the original paper but insufficient details actually provided to
%       explain what they actually do (it just says ``eliminate a random
%       subset of classes'' rather than providing details of for example
%       how many).  Some implementations omit this but it seems to be
%       important to practical performance.
%
% Outputs
%   R = sampled projection matrix
%   muX = Mean of inputs, to repeat projection, this needs subtracting from
%       the X first
%   U = projection, i.e. U = (X-muX)*R;
%
% 24/07/17

muX = mean(X,1);
X = bsxfun(@minus,X,muX);

D = size(X,2);

fOrder = randperm(D);

fGroups = reshape(fOrder(1:(M*floor(D/M))),M,[]);
fLeft = fOrder((M*floor(D/M)+1):end);

K = size(fGroups,2);

if prop_classes_eliminate~=0
    classes = unique(Y,'rows');
    nClasses = size(classes,1);
    
    n_classes_eliminate = floor(prop_classes_eliminate*nClasses);
    
    classLeaveGroups = manyRandPerms(nClasses,nClasses-n_classes_eliminate,size(fGroups,2))';
    classLeaveLeft = randperm(nClasses,nClasses-n_classes_eliminate);
    
    iClasses = cell(nClasses,1);
    for k=1:nClasses
        iClasses{k} = find(Y(:,k))';
    end
else
    % This could be a regression case so don't even try and find the
    % classes
    classLeaveGroups = ones(1,K);
    classLeaveLeft = 1;
    iClasses = {(1:size(X,1))};
end

R = zeros(D,D);
iUpTo = 1;

for n=1:K    
    iThis = [iClasses{classLeaveGroups(:,n)}];
    r = localRotation(X(iThis,fGroups(:,n)),prop_points_subsample);
    R((1+(n-1)*M):(n*M),iUpTo:(iUpTo+size(r,2)-1)) = r;
    iUpTo = iUpTo+size(r,2);
end

if ~isempty(fLeft)
    iThis = [iClasses{classLeaveLeft(:)}];
    r = localRotation(X(iThis,fLeft),prop_points_subsample);
    R((1+(K)*M):end,iUpTo:(iUpTo+size(r,2)-1)) = r;
end

R(fOrder,:) = R;

if nargout>2
    U = X*R;
end

end

function r = localRotation(x,p)
    
    iB = datasample((1:size(x,1)),round(size(x,1)*p));
    xB = x(iB,:);
    r = pcaLite(xB,false,false);

end
