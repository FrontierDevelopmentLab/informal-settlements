function kappa = cohenKappa(confusionMatrix)
% Calculates Cohen's kappa from a confusion matrix.  The confusion matrix
% can be generated from Matlab's confusionmat function.

propPreds = sum(confusionMatrix,1);
propPreds = propPreds/sum(propPreds);

propReal = sum(confusionMatrix,2);
propReal = propReal/sum(propReal);

p_e = sum(propPreds(:).*propReal(:));

p_o = sum(diag(confusionMatrix))/sum(confusionMatrix(:));

kappa = (p_o-p_e)./(1-p_e);