function [centered] = center_data(data)
%center_data scales the data to have zero mean along a given dimension
% dim - what dimension to normalise along
% data - the data that you want to center.
% 1 - rows
% 2 - columns
% >= 3 - other dimension

% Once the data is centered we return the centered matrix
% centered = (data - repmat(mean(data,dim), size(data,dim), dim)) ./ repmat(std(data,[],dim), size(data,dim), dim); 
meanrep= repmat(mean(data,2), [1 10]);
stdrep = repmat(std(data,[],2), [1 10]);
centered = (data - meanrep) ./stdrep; 
end
