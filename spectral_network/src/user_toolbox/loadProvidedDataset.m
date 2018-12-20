function [X,Y,bOrdinal] = loadProvidedDataset(name)
%loadProvidedDataset calls in datasets from Datasets folder
%
% [X,Y,bOrdinal] = loadProvidedDataset(name)
%
% Valid names are the names of any csv files in this folder (these will
% also be print to screen if you try and call an incorrect dataset).
% For details on the formatting then see loadCSVDataSet.m
%
% Tom Rainforth 04/08/15

loc = [regexprep(mfilename('fullpath'),mfilename,''),'..',filesep(),'..',filesep()];

class_loc = [loc 'Datasets' filesep() 'classification'];
reg_loc = [loc 'Datasets' filesep() 'regression'];
multi_loc = [loc 'Datasets' filesep() 'multi_output'];
class_dir = dir(class_loc); reg_dir = dir(reg_loc); multi_dir = dir(multi_loc);
validClass = cellfun(@(x) x(1:end-4), {class_dir(3:end).name},'UniformOutput',false);
validReg = cellfun(@(x) x(1:end-4), {reg_dir(3:end).name},'UniformOutput',false);
validMulti = cellfun(@(x) x(1:end-4), {multi_dir(3:end).name},'UniformOutput',false);

bClass = any(strcmp(name,validClass));
bReg = any(strcmp(name,validReg));
bMulti = any(strcmp(name,validMulti));

if ~bClass && ~bReg && ~bMulti
    disp('****** Valid Classification Names *******');
    disp(validClass')
    disp('****** Valid Regression Names *******');
    disp(validReg')
    disp('****** Valid Multi-Output Names *******');
    disp(validMulti')
    error('Invalid name, valid names shown above');
end

if bClass
    [X,Y,bOrdinal] = loadCSVDataSet([class_loc filesep() name '.csv']);
elseif bReg
    [X,Y,bOrdinal] = loadCSVDataSet([reg_loc filesep() name '.csv']);
else
    [X,Y,bOrdinal] = loadCSVDataSet([multi_loc filesep() name '.csv']);
end