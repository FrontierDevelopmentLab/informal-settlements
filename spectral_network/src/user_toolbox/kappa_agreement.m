function [kappas,avg_error] = kappa_agreement(tree_outputs,Y_true)
%[kappas,avg_error] = kappa_agreement(tree_outputs,Y_true)
% Calculates the kappa agreement (Cohen's kappa) for a selection of trees.  
% Classification only
%
% Inputs:
%       tree_outputs = Taken from predictFromCCF
%       Y_true = True classes
%
% Outputs:
%       kappas = matrix of kappa agreement for pairs of trees
%       avg_errors = matrix of average errors for pairs of trees
%
% 24/07/17

N = size(tree_outputs,1);
T = size(tree_outputs,2);
K = size(tree_outputs,3);

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
accs = NaN(T,1);
ps = NaN(T,K);

for n=1:T
    [~,tree_preds(:,n)] = max(squeeze(tree_outputs(:,n,:)),[],2);
    accs(n) = mean(tree_preds(:,n)~=Y_true);
    ps(n,:) = accumarray(tree_preds(:,n),ones(N,1),[K,1])';
    ps(n,:) = ps(n,:)/sum(ps(n,:));
end

kappas = NaN(T,T);
avg_error = NaN(T,T);

for n=1:T
    for m=1:T
        if n==m
            continue
        end
        avg_error(n,m) = 0.5*(accs(n)+accs(m));
        prob_agree = sum(tree_preds(:,n)==tree_preds(:,m))/N;
        ABC = sum(ps(n,:).*ps(m,:));
        kappas(n,m) = (prob_agree-ABC)./(1-ABC);
    end
end
        