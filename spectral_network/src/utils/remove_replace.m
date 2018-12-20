function [new_data] = remove_replace(data,replace_map)
% remove_replace takes data and removes certain values and replaces with 
% other values

%  data  - data from which we want to reduce
% replace_map - An n_classes x n_replace matrix of the mapping from one
% class to the other. For example let's say that we want to map the values
% [3  2 1] to [1 0 0] then replace_map = [3 2 1; 1 0 0]

% returns updated data matrix
[r,c] = size(replace_map);

for ii= 1:c
   data(data == replace_map(1,ii)) = replace_map(2,ii) ;
end

new_data = data;
end

