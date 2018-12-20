function classifyimagetest(CCF,country,source,city,testcity,type,method, ntrees, server)
% Classify image function takes in a set of file names, loads the relevant 
% .tif image and loads the relevant ccfs pre-trained model and then
% performs the prediction and then saves the prediction mask
%
% The arguments are:
% country - str of country name 
% source - str of either 'S2' or 'DG'
% city - str of city name
% type - str either 'test' or 'training'
% method - str either 'spec2imf' or 'spec2mat'

%   
    if strcmp(city,testcity)
        disp('Classify on same train set');
    else
        return
    end
    if ~exist('server','var')
        server = '';
    end
        
    load_inf = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
    gt = '_ground_truth';
    filetype='.tif';
    extgt = strcat(city,gt,filetype);
    ext = strcat(city,filetype);
    fgt = fullfile(load_inf,country,city,source,type,extgt);
    ft  =fullfile(load_inf,country,city,source,type,ext);
%   image_ground_truth=  double(imread(fgt));
    image_test = double(imread(ft));
    lfname = strcat(server,'model/',method,'/pre_trained_',testcity,'_with_',num2str(ntrees),'trees.mat');
    fname = strcat(country,'/',city,'/images/','pred_with_',testcity,'_image_mask.mat');
    %% Load pretrained model

%     CCF = load(lfname);
    % Separate into to each column of image, which represents nrows x 10 matrix. 
    image_array = image_test;
    [n m p]  = size(image_test);
    % pre-allocating memory
    image_mask = double(zeros(n,m));
    parfor ii = 1:m
%         disp(['Creating mask prediction of column ' num2str(ii)])
        temp = image_array(:,ii,:);
        temp = reshape(temp, [],10);
        temp = transpose(temp);
        temp = (temp - repmat(mean(temp), size(temp,1), 1)) ./ repmat(std(temp), size(temp,1), 1);
        temp = transpose(temp);
        prediction = predictFromCCF(CCF,temp);
        image_mask(:,ii)=prediction;
    end

    disp('Saving classified image...');
    filename = strcat(server,'predictions/',method,fname);
    save(filename, 'image_mask');

end

