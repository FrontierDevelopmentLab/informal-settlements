function [f1_weighted_mean, f1_raw] = f1score(ypred,ytrue)

preds = unique(ypred);
trues = unique(ytrue);
vals = union(preds,trues);
b_pred = bsxfun(@eq,ypred,vals');
b_true = bsxfun(@eq,ytrue,vals');
tp = sum(b_pred & b_true);
all_true = sum(b_true);
all_pred = sum(b_pred);
f1_raw = 2*tp./(all_true+all_pred);
f1_weighted_mean = sum(f1_raw.*all_true/(sum(all_true)));

end