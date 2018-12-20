function predictinf2spec(CCF,country,source,city,testcity,type,method,multiclass, class_map,classes_remove,ntrees,server, predict_data)

% Performs a prediction on the given test set and saves the metrics: IOU,
% classification percentage of informal, classification percentage of 
% environment. 

% Inputs
% multiclass - int [1 = true, 0 = false] If multiclass, will relabel all labels to be either 
%               1 for informal or 0 for everything else. Specified by
%               class map
% country - str of country name 
% source - str of either 'S2' or 'DG'
% city - str of city name
% testcity - the pre-trained citymodel used for testing
% type - str either 'test' or 'training'
% method     - str either 'spec2imf' or 'spec2mat'
% class_map - array that represents a mapping from the original class
% values to new class values i.e class_map = [3 2 1 5; 0 0 0 1];
% classes_remove - array of classes to be removed.
% ntrees - int- used for retriving the right model
% predict - bool - to avoid testing and training on the same data, when
% this flag is activated it automatically performs the prediction and 
% will avoid the prediction during the second function call in run
% experiments. 
% predict_data - Matrix - Array of data from splitting the training and
% test set. removes the need to have to reload dataset when training on the
% same model. 

if strcmp(testcity,city)
    if exist('predict_data', 'var')
        disp(' Validating trained city on trained city.');
        predict = 1;
    else
        predict = 0;
    end
else
    predict = 0;
end
if ~exist('multiclass','var')
    multiclass = 0;
end
if ~exist('classes_remove','var')
    classes_remove = [];
end
if ~exist('class_map','var')
    class_map = [];
end
if ~exist('server','var')
    server='';
end

if ~exist('predict_data', 'var')
    predict_data = 0;
end


if predict
    spectrum = predict_data;
    baseModel = strcat(server,'model/');
    fload = strcat(baseModel,method,'/pre_trained_',testcity,'_with_',num2str(ntrees),'trees.mat');
%     CCF = load(fload);

else
    baseModel = strcat(server,'model/');
    fload = strcat(baseModel,method,'/pre_trained_',testcity,'_with_',num2str(ntrees),'trees.mat');
%     CCF = load(fload);
    if strcmp(testcity,city)
        disp(' Warning : Cannot test and train on the same data points')
        return
    end
    baseTest = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
    endf = strcat(city,'_ground_truth.mat');
    ftest = strcat(baseTest,country,'/',city,'/',source,'/',type,'/',endf);
    alldata = load(ftest);
    alldata = alldata.ground_truth;
    spectrum = alldata(:,1:10);
    classes = alldata(:,11);

    %% Normalise data to be zero mean and variance 1

    spectrum = center_data(spectrum);

    %% Transpose and concat arrays into one array for testing
    spectrum = [spectrum, classes];

    if multiclass==1
        %% Remove unecessary classes
    %     classes_remove = [4 6];
        spectrum = remove_classes(spectrum, classes_remove, 11);
        %% Replace classes
        spectrum = remove_replace(spectrum,class_map);
    end

end

%% Test model
disp(['CCF prediction on ', city]);
YpredCCF = predictFromCCF(CCF,spectrum(:,1:10));
YTest = spectrum(:,11);
disp(['CCF Test missclassification rate (lower better) ' num2str(100*(1-mean(mean(YTest==(YpredCCF))))) '%']);
f1=YTest == 1;
classification1 = sum(sum(YTest(f1) == YpredCCF(f1))) / sum(sum(f1)) ;
f0=YTest == 0;
classification0 = sum(sum(YTest(f0) == YpredCCF(f0))) / sum(sum(f0));

metrics = cell(2,2);
metrics{1,1}='Informal pixel classification : ';
metrics{1,2}= classification1;
metrics{2,1}= 'Environment pixel classification : ' ;
metrics{2,2} = classification0;


dirname = strcat(server,'predictions/',country,'/',city);
mkdir_if_not_exist(dirname);
fmetric = strcat(dirname,'/prediction_with_',testcity,'_metrics.dat');
fileID = fopen(fmetric,'w');
formatSpec = '%s %.5f \n';
[nrows,ncols] = size(metrics);
for row = 1:nrows
    fprintf(fileID,formatSpec,metrics{row,:});
end

fclose(fileID);

end

