function [cleaned_data] = remove_classes(data,class_names, class_index)
% remove_classes :
% data - the data which contains data and class labels
%  class_names - matrix of classes to remove from data matrix
% class index - what column are the classes stored in

n_class = length(class_names);
for ii=1:n_class
    index = data(:,class_index) == class_names(ii);
    data(index,:) = [];
end
cleaned_data = data ;
end

