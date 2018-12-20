function load_city(country,source,city,type,split,multiclass, class_map,classes_remove,server)

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
end