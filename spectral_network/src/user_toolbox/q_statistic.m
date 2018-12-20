function [Q,avg_error] = q_statistic(tree_outputs,Y_true)
%[Q,avg_error] = q_statistic(tree_outputs,Y_true)
% Calculates the q statistic for a selection of trees.  Classification only
%
% Inputs:
%       tree_outputs = Taken from predictFromCCF
%       Y_true = True classes
%
% Outputs:
%       Q = matrix of q statistics for pairs of trees
%       avg_errors = matrix of average errors for pairs of trees
%
% 24/07/17

N = size(tree_outputs,1);
T = size(tree_outputs,2);

if T>1000
    warning(['n_sample all option will try to produce ' num2str(T.^2) ' kappas. This could be very slow!']);
    if size(T,2)>10000
        error('Too many tree outputs, pass in less');
    end
end

if size(Y_true,2)>1
    [~,Y_true] = max(Y_true,[],2);
end

tree_preds = NaN(N,T);
bCorr = NaN(N,T);
accs = NaN(T,1);

for n=1:T
    [~,tree_preds(:,n)] = max(squeeze(tree_outputs(:,n,:)),[],2);
    bCorr(:,n) = tree_preds(:,n)~=Y_true;
    accs(n) = mean(bCorr(:,n));
end

Q = NaN(T,T);
avg_error = NaN(T,T);

for n=1:T
    for m=1:T
        if n==m
            continue
        end
        avg_error(n,m) = 0.5*(accs(n)+accs(m));
        N00 = sum(bCorr(:,n) & bCorr(:,m));
        N10 = sum(bCorr(:,n) & ~bCorr(:,m));
        N01 = sum(~bCorr(:,n) & bCorr(:,m));
        N11 = sum(~bCorr(:,n) & ~bCorr(:,m));
        Q(n,m) = (N11*N00-N01*N10)/(N11*N00+N01*N10);
    end
end
        