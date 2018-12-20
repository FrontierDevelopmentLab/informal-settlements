function bVar = queryIfColumnsVary(X,tol)
% Function that says whether columns are constant or not
bVar = max(abs(diff(X(1:min(5,size(X,1)),:),[],1)),[],1)>tol;
bVar(~bVar) = max(abs(diff(X(:,~bVar),[],1)),[],1)>tol;
end