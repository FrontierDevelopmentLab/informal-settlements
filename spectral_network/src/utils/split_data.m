function [training,validation] = split_data(data,split)
% splits the training and testing data, depending on percentage split. 

% data - A matrix of training data
% split - How to split the data

% Outputs

% training -  data split according to training split
% validation - data split according to validation split


[q r] = size(data);
split_train = floor(split*q); 
split_val = floor((1-split)*q);
training = transpose(data(1:split_train,:));
validation = transpose(data(split_train+1:end,:));

end

