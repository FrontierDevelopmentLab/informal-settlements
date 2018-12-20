function [A, B, U, V, r] = componentAnalysis(X,Y,processes,epsilon)
%componentAnalysis  Carries out a series of specified component analyses
%
% [A, B, r, U, V] = componentAnalysis(X,Y,processes,epsilon)
%
% Carries out a a section of component analyses on X and Y to produce a
% projection matrix projMat which maps X to its components.  Valid
% projectiosn are CCA, PCA, CCA-classwise, original axes and random
% rotation
%
% Inputs:
%          X = Input features, must be numeric.  Each row is a datapoint
%              and each column a feature (categoricals should have been
%              converted to their binary expansion)
%          Y = Output matrix.  For classification then must be in 1-of-K
%              expansion format
%  processes = Component analyses to carry out. This should be a structure
%              of the form of the projections field in optionsClassCCF
%    epsilon = Tolerance parameter for rank reduction of QR decompositions
%
% Ouputs:
%          A = Projection matrix for X
%          B = Projection matrix for Y
%          U = X Components = (X-mean(X))*A
%          V = Y Components = (Y-mean(Y))*B
%          r = Canonical correlations, only provided for CCA process
%
% Tom Rainforth 25/06/17

% Sample projections to use if some set to be probabilistically used
probs = struct2array(processes);
bToSample = (probs>0)&(probs<1);
if any(bToSample)
    probs(~bToSample) = 0;
    cumprobs = cumsum(probs)/sum(probs);
    iSampled = sum(rand>cumprobs)+1;
    process_fields = fields(processes);
    iToSample = find(bToSample);
    for n=1:numel(iToSample)
        processes.(process_fields{iToSample(n)}) = false;
    end
    processes.(process_fields{iSampled}) = true;    
end

% Eliminate any columns that don't vary.  We will add these back into the
% projection matrices at the end
bXVaries = queryIfColumnsVary(X,1e-12);
bYvaries = queryIfColumnsVary(Y,1e-12);
nXorg = numel(bXVaries);
nYorg = numel(bYvaries);

if ~any(bXVaries) || ~any(bYvaries)
    % One of X or Y doesn't vary so component analysis fails.  Return
    % projection corresponding to first columns of X and Y
    A = [1;zeros(nXorg-1,1)];
    if nargout>1
        B = [1;zeros(nYorg-1,1)];
        if nargout>2
            U = X(:,1);
            V = Y(:,1);
            r = 0;
        end
    end
    return
end

X = X(:,bXVaries);
Y = Y(:,bYvaries);

% Checks and sizes
[x1,x2] = size(X);
if size(Y,1) ~= x1
    error('Input sizes do not match');
elseif x1 == 1
    error('Cannot carry out component analysis with only one point');
end
K = size(Y,2);

% Subtraction of the mean is common to the process of calculating the
% projection matrices for both CCA and PCA but for computational
% effificently we don't make this translation when actually applying the
% projections to choose the splits as it is the same effect on all points.
% In other words, we don't split in canonical component space exactly, but
% in a constant translation of this space.
muX = sum(X,1)/size(X,1);
muY = sum(Y,1)/size(Y,1);
X = bsxfun(@minus,X,muX);
Y = bsxfun(@minus,Y,muY);

% Initialize the project matrices
projMat = NaN(size(X,2),0);
yprojMat = NaN(size(Y,2),0);
r = [];

if processes.Original
    % Include original axes in projection
    projMat = [projMat,eye(x2)];    
end

if processes.Random   
    % Random projection matrix
    projMat = [projMat,randomRotation(x2)];    
end

if processes.PCA    
    % PCA projection
    pcaCoeff = pcaLite(X);
    projMat = [projMat,pcaCoeff];    
end

if processes.CCA || processes.CCAclasswise
    % CCA based projections
    
    % These require QR decomposition of X
    [q1,r1,p1] = qr(X,0);
    % Reduce to full rank within some tolerance
    if isempty(r1)
        rankX = 0;
    else
        rankX = sum(abs(diag(r1)) >= (epsilon*abs(r1(1))));
    end
    if rankX == 0
        %warning('X doesnt vary so component analysis fails');
        A = [1;zeros(nXorg-1,1)];
        if nargout>1
            B = [1;zeros(nYorg-1,1)];
            if nargout>2
               U = X(:,1);
               V = Y(:,1);
               r = 0;
            end
        end
        return
    elseif rankX < x2
        q1 = q1(:,1:rankX); 
        r1 = r1(1:rankX,1:rankX);
    end
    
    if processes.CCA 
        
        % This code is a reduction of the function canoncorr.  This
        % method is explained in the supplementary material
        
        [q2,r2,p2] = qr(Y,0);
        % Reduce to full rank within some tolerance
        if isempty(r2)
            rankY = 0;
        else
            rankY = sum(abs(diag(r2)) >= (epsilon*abs(r2(1))));
        end
        if rankY == 0
            %warning('Y doesnt vary so component analysis fails');
            A = [1;zeros(nXorg-1,1)];
            if nargout>1
                B = [1;zeros(nYorg-1,1)];
                if nargout>2
                    U = X(:,1);
                    V = Y(:,1);
                    r = 0;
                end
            end
            return
        elseif rankY < K
            q2 = q2(:,1:rankY); 
        end
        
        % Solve CCA using the decompositions, taking care to use minimal
        % complexity orientation for SVD.  Note the two calculations are
        % equivalent except in computational complexity
        
        d = min(rankX,rankY);
        if rankX>=rankY
            [L,D,M] = svd(q1' * q2,0);
        else
            [M,D,L] = svd(q2' * q1,0);
        end
        locProj = r1 \ L(:,1:d) * sqrt(x1-1);
                
        % Put coefficients back to their full size and their correct order
        locProj(p1,:) = [locProj; zeros(x2-rankX,d)];
        projMat = [projMat,locProj];
        
        if nargout>1
            % If requested also return projection for Y
            r2 = r2(1:rankY,1:rankY);
            locyProj = r2 \ M(:,1:d) * sqrt(x1-1);
            locyProj(p2,:) = [locyProj; zeros(K-rankY,d)];
            yprojMat = [yprojMat,locyProj];
        end
        
        if nargout>2
            r = min(max(diag(D(:,1:d))', 0), 1); %#ok<UDIM>
        end
    end
    
    if processes.CCAclasswise
        % Consider each output in an in / out fashion to generate a set
        % of K projections.
        
        for k=1:K
            % Solve CCA using the X decomposition and the fact that the
            % corresponding Y decomposition is itself
            [L,~,~] = svd(q1' * Y(:,k),0);
            locProj = r1 \ L(:,1) * sqrt(x1-1);
            locProj(p1,:) = [locProj; zeros(x2-rankX,1)];
            projMat = [projMat,locProj];       %#ok<AGROW>
        end
    end
    
end

% Normalize the projection matrices.  This ensures that the later tests for
% close points are triggered appropriately and is useful for
% interpretability.
projMat = bsxfun(@rdivide,projMat,sqrt(sum(projMat.^2,1)));

if nargout>2
    % Note that as in general only a projection matrix is given, we need to
    % add the mean back to be consistent with general use.  This equates to
    % addition of a constant term to each column in U
    U = X*projMat;
    if nargout>3
        V = Y*yprojMat;
    end
end

% Finally, add back in the empty rows in the projection matrix for the
% things which didn't vary

A = zeros(nXorg,size(projMat,2));
A(bXVaries,:) = projMat;
if nargout>1 
    B = zeros(nYorg,size(yprojMat,2));
    B(bYvaries,:) = yprojMat;
end

end