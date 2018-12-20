function meanIOU2(country,source,city,testcity,type,method, server)
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
%     if strcmp(city,testcity)
%         disp('Cannot classify on same train set');
%         return
%     end
    if ~exist('server','var')
        server = '';
    end
    nclasses =2;
    load_inf = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
    gt = '_ground_truth';
    filetype='.tif';
    extgt = strcat(city,gt,filetype);
    ext = strcat(city,filetype);
    fgt = fullfile(load_inf,country,city,source,type,extgt);
    ft  =fullfile(load_inf,country,city,source,type,ext);
    image_ground_truth=  double(imread(fgt)); 
    
%     server = '/Users/bradley/Documents/Projects/Team2_FDL/synthesis-generate-spectrum/';
    base = strcat(server,'predictions/');
    
    full = strcat(base,country,'/',city,'/images/');
    img_mas = strcat(full,'pred_with_',testcity,'_image_mask.mat');
    image_mask = load(img_mas);
    image_pred = image_mask.image_mask;
    
    num = nclasses; 
    
    confcounts = zeros(num);
    count=0;
    tic;

    % ground truth label file
%         [gtim,map] = imread(gtfile);    
    gtim = image_ground_truth;

    % results file
%         resfile = sprintf(VOCopts.seg.clsrespath,id,VOCopts.testset,imname);
%         [resim,map] = imread(resfile);
    resim = image_pred;

    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>nclasses)
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,VOCopts.nclasses);
    end

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end

    %pixel locations to include in computation
    locs = gtim<255;

    % joint histogram
    sumim = 1+gtim+resim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);

    % confusion matrix - first index is true label, second is inferred label
    %conf = zeros(num);
    conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
    rawcounts = confcounts;

    % Percentage correct labels measure is no longer being used.  Uncomment if
    % you wish to see it anyway
    %overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
    %fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

    accuracies = zeros(nclasses,1);
    fprintf('Accuracy for each class (intersection/union measure)\n');
    classes = {'background','infromal'};
    for j=1:num

       gtj=sum(confcounts(j,:));
       resj=sum(confcounts(:,j));
       gtjresj=confcounts(j,j);
       % The accuracy is: true positive / (true positive + false positive + false negative) 
       % which is equivalent to the following percentage:
       accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);   

%        clname = 'background';
%        if (j>1), clname = VOCopts.classes{j-1};end;
       clname = classes{j};
       fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
    end
 accuracies = accuracies(1:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
fprintf('Average accuracy: %6.3f%%\n',avacc);
metrics = cell(3,2);
metrics{1,1}='Background IOU :';
metrics{1,2} = accuracies(1);
metrics{1,1}='Informal IOU :';
metrics{2,2} = accuracies(2);
metrics{3,1}= ' Mean Accuracy';
metrics{3,2}= avacc;

dirname = strcat(server,'predictions/',country,'/',city);
mkdir_if_not_exist(dirname);
fmetric = strcat(dirname,'/','meanIOU_ontrain_prediction_with_',testcity,'_metrics.dat');
fileID = fopen(fmetric,'w');
formatSpec = '%s %.5f \n';
[nrows,ncols] = size(metrics);
for row = 1:nrows
    fprintf(fileID,formatSpec,metrics{row,:});
end

fclose(fileID);
end

