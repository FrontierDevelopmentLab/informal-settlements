% Script for adding the required Matlab paths.  Should be run once at the
% start of each session to add all required folders to the matlab path,
% does not need running after

this_path = regexprep(mfilename('fullpath'),mfilename,'');
req_folders = {'Datasets','example_scripts','src'};
mypath = path;

for n=1:numel(req_folders)
    req_path = genpath([this_path filesep() req_folders{n}]);
    if ~contains(mypath,req_path)
        addpath(req_path);
    end
end