function [CCF] = inform2spec(country,source,city,type,method,split,multiclass, class_map,classes_remove, ntrees,server,train_test_on_same)
%Runs the training for CCFs on spectral data to informal. 
% Saves the trained model in the country specific model folder. 

% Inputs
% multiclass - int [1 = true, 0 = false] If multiclass, will relabel all labels to be either 
%               1 for informal or 0 for everything else. Specified by
%               class map
% split - double in range [0,1] Percentage of dataset to use for training
% and validation
% country - str of country name 
% source - str of either 'S2' or 'DG'
% city - str of city name
% type - str either 'test' or 'training'
% method   - str either 'spec2imf' or 'spec2mat'
% class_map - array that represents a mapping from the original class
% values to new class values i.e class_map = [3 2 1 5; 0 0 0 1];
% classes_remove - array of classes to be removed.
% ntrees - number of trees to use for constructing ccf forest 
% server - str - path to where ever the source code is being run from
% predict - bool - to avoid testing and training on the same data, when
% this flag is activated it automatically performs the prediction and 
% will avoid the prediction during the second function call in run
% experiments. 
if ~exist('multiclass','var')
    multiclass = 0;
end
if ~exist('classes_remove','var')
    classes_remove = [];
end
if ~exist('ntrees','var')
    ntrees = 50;
end
if ~exist('class_map','var')
    class_map = [];
end
if ~exist('server','var')
    server='';
end
 
if ~exist('train_test_on_same','var')
    train_test_on_same = 0;
end
base = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
endn = strcat(city,'_ground_truth.mat');
lfname  = strcat(base,'/',country,'/',city,'/',source,'/',type,'/',endn);
alldata = load(lfname);
alldata = alldata.ground_truth;

classes = alldata(:,11);
spectrum = alldata(:,1:10);
%% Balance the dataset
disp(' Balancing the data set');
NumOnes = sum(classes == 1);
NumZeros = sum(classes ==0);

if NumOnes > NumZeros
    diff = NumOnes - NumZeros;
    NumOnes = NumOnes-diff;
elseif NumOnes < NumZeros
    diff = NumZeros - NumOnes;
    NumZeros = NumZeros-diff;
end


total_data = NumOnes + NumZeros;

indx1 = classes ==1;
indx0 = classes ==0;
spectrum_ones = spectrum(indx1,:);
spectrum_zeros = spectrum(indx0,:);

index = randsample(1:NumOnes,floor(NumOnes*split),true);
spectrum_ones_train = spectrum_ones(index',:);
spectrum_ones(index',:)=[];
spectrum_ones_vald = spectrum_ones;
index = randsample(1:NumZeros,floor(NumZeros*split),true);
spectrum_zeros_train= spectrum_zeros(index',:);
spectrum_zeros(index',:)=[];
spectrum_zeros_vald = spectrum_zeros(1:size(spectrum_ones_vald,1),:);


%% splitting the testing and training 
disp(['Splitting the data into training ~ ' num2str(split*100) '% validation ~' num2str((1-split)*100) '%']);

train_spectrum = vertcat(spectrum_ones_train,spectrum_zeros_train);
train_classes =  vertcat(ones(size(spectrum_ones_train,1),1),zeros(size(spectrum_zeros_train,1),1));
validation_spectrum = vertcat(spectrum_ones_vald,spectrum_zeros_vald);
validation_classes =  vertcat(ones(size(spectrum_ones_vald,1),1),zeros(size(spectrum_zeros_vald,1),1));

%% Normalise data to be zero mean and variance 1

train_spectrum = center_data(train_spectrum);
validation_spectrum =  center_data(validation_spectrum);

%% Transpose and concat arrays into one array for trianing
train_spectrum = [train_spectrum, train_classes];
validation_spectrum = [validation_spectrum, validation_classes];

if multiclass==1
    %% Remove unecessary classes
    train_spectrum = remove_classes(train_spectrum, classes_remove, 11);
    validation_spectrum = remove_classes(validation_spectrum, classes_remove, 11);
    %% Replace classes
    train_spectrum = remove_replace(train_spectrum,class_map);
    validation_spectrum = remove_replace(validation_spectrum,class_map);
end
%% Train model
disp('Training CCF')
% CCF = genCCF(ntrees, train_spectrum(:,1:10),train_spectrum(:,11));
% Uncomment if for ccfs

% the below performs training with random forests. 
obj = optionsClassCCF.defaultOptionsRF;
CCF = genCCF(ntrees, train_spectrum(:,1:10),train_spectrum(:,11),false,obj);
%% save model
% disp(' Saving pretrained model ');
% dirname = strcat(server,'model/',method);
% mkdir_if_not_exist(dirname)
% fsave = strcat(dirname,'/','pre_trained_',city,'_with_',num2str(ntrees),'trees.mat');
% save(fsave,'-struct','CCF', '-v7.3','-nocompression');

void = 0;
if train_test_on_same
    predictinf2spec(CCF,country,source,city,city,type,method,multiclass,void,void,ntrees,server,validation_spectrum);
end
% YpredCCF = predictFromCCF(CCF,validation_spectrum(:,1:10));
% YTest = validation_spectrum(:,11);

end

