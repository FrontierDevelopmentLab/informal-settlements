function [ground_truth] = concat_all_binary(country,cities,source,type,server,ground_truth)
% Takes all of the binary .mat files produced for each city and 
% concatenates them into one large informal formal matrix.
    if ~exist('server','var')
        server='';
    end
    if ~exist('ground_truth','var')
        ground_truth = [];
    end
    load_inf = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
    gt = '_ground_truth';
    filetype='.mat';
    for jj=1:length(cities)
        extgt = strcat(cities{jj},gt,filetype);
        fgt = fullfile(load_inf,country,cities{jj},source,type,extgt);
        image_ground_truth=  double(imread(fgt));
        ground_truth = vertcat(allm,image_ground_truth);
    end
   
end
