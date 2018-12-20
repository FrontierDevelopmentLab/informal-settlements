function train_spectral(dataset,method,type,fext, ntrees,balance,classes,server)
    if ~exist('ntrees','var')
        ntrees = 50;
    end
    if ~exist('server','var')
        server='';
    end
    %% Load data

    fall = strcat(server,'Training_sets_and_ground_truth/material_classification/',dataset,'/',type,'/',dataset,fext);
    all_spectrums = load(fall);
    % all_spectrums= all_spectrums.afrobarometer ;
    fshings  = strcat(server,'Training_sets_and_ground_truth/material_classification/',dataset,'/',type,'/','shingles',fext);
    true_shingles=  load(fshings);
    formatSpec = '%f';


    %% Center the data to have mean 0 and variance 1.
    class = all_spectrums(:,11);
    spectrums = center_data(all_spectrums(:,1:10));
    true_shingles(:,1:10) = center_data(true_shingles(:,1:10));
    data_centered = [spectrums, class];

    %% Renormalize sampeld_shingals
    [n,m] = size(true_shingles);
    index = randsample(1:n, 10, true);
    sampled_shingles = true_shingles(index,:);

    if balance
        %% Sepearate classes data set.
        % class 2 has the lowest amount of data 
        ind0 = data_centered(:,11) ==0;
        ind1 = data_centered(:,11) ==1;
        ind2 = data_centered(:,11) ==2;
        ind3 = data_centered(:,11) ==3;
        ind4 = data_centered(:,11) ==4;
        ind5 = data_centered(:,11) ==5;
        ind6 = data_centered(:,11) ==6;
        class_0 = data_centered(ind0,:);
        class_1 = data_centered(ind1,:);
        class_2 = data_centered(ind2,:);
        class_3 = data_centered(ind3,:);
        class_4 = data_centered(ind4,:);
        class_5 = data_centered(ind5,:);
        class_6 = data_centered(ind6,:);

        %% add lab data to class 3
        class_3 = vertcat(class_3, sampled_shingles) ;
        class_3(class_3 == 2) = 3;

        %% Creating a balanced training and validation set
        [a0,b0] = size(class_0);
        [a,b] = size(class_1);
        [c,d] = size(class_2);
        [e,f] = size(class_3);
        [g,h] = size(class_4);
        [i,j] = size(class_5);
        [k,l] = size(class_6);
        % train on 80% of the size of the smallest class. Validate on the remaining
        % 20%. 
        datapoints = e;

        train_size = floor(0.8*datapoints);

        val_size = datapoints - train_size;

        % balancing the datasets
        index = randsample(1:a0, datapoints, true);
        reduced_c0 =class_0(index,:);
        index = randsample(1:a, datapoints, true);
        reduced_c1 =class_1(index,:);
        index = randsample(1:c, datapoints, true);
        reduced_c2 =class_2(index,:);
        index = randsample(1:e, datapoints, true);
        reduced_c3 =class_3(index,:);
        index = randsample(1:g, datapoints, true);
        reduced_c4 =class_4(index,:);
        index = randsample(1:i, datapoints, true);
        reduced_c5 =class_5(index,:);
        index = randsample(1:k, datapoints, true);
        reduced_c6 =class_6(index,:);

        all_data_training  = vertcat(reduced_c0,reduced_c1,reduced_c2,reduced_c3, reduced_c4, reduced_c5,reduced_c6);
        % split in to training and validation
        index = randsample(1:datapoints, train_size, true);
        c0_train = reduced_c0(index,:);
        reduced_c0(index,:) = [];
        c0_test = reduced_c0;
        index = randsample(1:datapoints, train_size, true);
        c1_train = reduced_c1(index,:);
        reduced_c1(index,:) = [];
        c1_test = reduced_c1;
        index = randsample(1:datapoints, train_size, true);
        c2_train =reduced_c2(index,:);
        reduced_c2(index,:) = [];
        c2_test = reduced_c2;
        index = randsample(1:datapoints, train_size, true);
        c3_train =reduced_c3(index,:);
        reduced_c3(index,:) = [];
        c3_test = reduced_c3;
        index = randsample(1:datapoints, train_size, true);
        c4_train =reduced_c4(index,:);
        reduced_c4(index,:) = [];
        c4_test = reduced_c4;
        index = randsample(1:datapoints, train_size, true);
        c5_train =reduced_c5(index,:);
        reduced_c5(index,:) = [];
        c5_test = reduced_c5;
        index = randsample(1:datapoints, train_size, true);
        c6_train =reduced_c6(index,:);
        reduced_c6(index,:) = [];
        c6_test = reduced_c6;

        training_data = vertcat(c0_train, c1_train, c2_train, c3_train, c4_train, c5_train, c6_train);
        test_data = vertcat(c0_test, c1_test, c2_test, c3_test, c4_test, c5_test, c6_test);
%         training_data = vertcat(c0_train, c1_train, c2_train, c3_train, c4_train);
%         test_data = vertcat(c0_test, c1_test, c2_test, c3_test, c4_test);
    else
        %% Sepearate classes data set.
        % class 2 has the lowest amount of data 
        ind0 = data_centered(:,11) ==0;
        ind1 = data_centered(:,11) ==1;
        ind2 = data_centered(:,11) ==2;
        ind4 = data_centered(:,11) ==4;

        class_0 = data_centered(ind0,:);
        class_1 = data_centered(ind1,:);
        class_2 = data_centered(ind2,:);
        class_4 = data_centered(ind4,:);
        %% so that class labels are 0,1,2,3
        class_4(class_4 == 4) = 3;
        %% Creating a balanced training and validation set
        [a0,b0] = size(class_0);
        [a,b] = size(class_1);
        [c,d] = size(class_2);
%         [e,f] = size(class_3);
        [g,h] = size(class_4);
%         [i,j] = size(class_5);
%         [k,l] = size(class_6);
        % train on 80% of the size of the smallest class. Validate on the remaining
        % 20%. 
        datapoints = g;

        train_size = floor(0.8*datapoints);

        val_size = datapoints - train_size;

        % balancing the datasets
        index = randsample(1:a0, datapoints, true);
        reduced_c0 =class_0(index,:);
        index = randsample(1:a, datapoints, true);
        reduced_c1 =class_1(index,:);
        index = randsample(1:c, datapoints, true);
        reduced_c2 =class_2(index,:);
        index = randsample(1:g, datapoints, true);
        

        all_data_training  = vertcat(reduced_c0,reduced_c1,reduced_c2, reduced_c4);
        % split in to training and validation
        index = randsample(1:datapoints, train_size, true);
        c0_train = reduced_c0(index,:);
        reduced_c0(index,:) = [];
        c0_test = reduced_c0;
        
        index = randsample(1:datapoints, train_size, true);
        c1_train = reduced_c1(index,:);
        reduced_c1(index,:) = [];
        c1_test = reduced_c1;
        
        index = randsample(1:datapoints, train_size, true);
        c2_train =reduced_c2(index,:);
        reduced_c2(index,:) = [];
        c2_test = reduced_c2;
       
        index = randsample(1:datapoints, train_size, true);
        c4_train =reduced_c4(index,:);
        reduced_c4(index,:) = [];
        c4_test = reduced_c4;

    %     training_data = vertcat(c0_train, c1_train, c2_train, c3_train, c4_train, c5_train, c6_train);
    %     test_data = vertcat(c0_test, c1_test, c2_test, c3_test, c4_test, c5_test, c6_test);
        training_data = vertcat(c0_train, c1_train, c2_train,  c4_train);
        test_data = vertcat(c0_test, c1_test, c2_test, c4_test);
    end
    %% Train model
    disp('Training CCF')
    CCF = genCCF(ntrees, training_data(:,1:10),training_data(:,11)); 
    YpredCCF = predictFromCCF(CCF,test_data(:,1:10));
    YTest = test_data(:,11);
    disp(['CCF Test missclassification rate (lower better) ' num2str(100*(1-sum(sum(YTest==(YpredCCF))))) '%']);


    % Print classification rates per class

    for ii = 1:classes
        f1=YTest == ii-1;
        disp([' The classification rate for class ' num2str(ii-1) ' is : ' num2str(sum(sum(YTest(f1) == YpredCCF(f1))) / sum(sum(f1))) '%']);
    end


    %% Saving pre-trained model 
    disp(' Saving pre-trained model');
    dirname = strcat(server,'model/',method);
    mkdir_if_not_exist(dirname);
    fname= strcat(dirname, '/pre_trained_on_',dataset,'_with_', num2str(ntrees),'trees.mat');
    save(fname,'-struct','CCF');
