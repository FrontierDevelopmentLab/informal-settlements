%% Defining experiment params 
testcities = {'Mumbai','Capetownsmall','Lower','Kibera','Kianda', 'AlGeneina','ElDaein','Mokako','Medellin'};
kenya.name = 'Kenya';
kenya.cities = {'Lower','Kibera' ,'Kianda'};
kenya.testcities = testcities;
kenya.classmaps = {[],[],[]};
kenya.multiclass = {0,0,0};
kenya.classremove = {[], [], []};
kenya.ccf = {};

sudan.name = 'Sudan';
sudan.cities = {'AlGeneina','ElDaein'};
sudan.testcities =testcities;
sudan.classmaps = {[];[]};
sudan.multiclass = {0,0};
sudan.classremove = {[];[]};
sudan.ccf = {};
% 
india.name = 'India';
india.cities = {'Mumbai'};
india.testcities = testcities;
india.classmaps = {[]};
india.multiclass = {0};
india.classremove = {[]};
india.ccf = {};

southafrica.name = 'Southafrica';
southafrica.cities = {'Capetownsmall'};
southafrica.testcities =testcities;
southafrica.classmaps = {[],[]};
southafrica.multiclass = {0,0};
southafrica.classremove = {[],[]};
southafrica.ccf = {};

nigeria.name = 'Nigeria';
nigeria.cities = {'Mokako'};
nigeria.testcities =testcities;
nigeria.classmaps = {[]};
nigeria.multiclass = {0};
nigeria.classremove = {[]};
nigeria.ccf = {};

colombia.name = 'Colombia';
colombia.cities = {'Medellin'};
colombia.testcities =testcities;
colombia.classmaps = {[]};
colombia.multiclass = {0};
colombia.classremove = {[]};
colombia.ccf = {};

data.countries = {southafrica, nigeria, kenya, india, sudan,colombia};
% global constants
source = 'S2';
type= 'training';
split = 0.8;
% as we are just doing binary pred
classes = 2;
% method for all 
method= 'spec2inf';
lbl = {'Environment', 'Informal'};
cmap = [0 1 ; 0 1; 0 1];
server = '/Users/bradley/Documents/Projects/Team2_FDL/synthesis-generate-spectrum/ccfs/';
nCountries= length(data.countries);
%% Set the things that you would like to do
create_binary_mask =0;
train_model =  1;
test_model =1;
classify_image =1;
create_image =1;
calc_meanIOU = 1;
create_all =0;
ntrees = 15;
predict = 1; % Train and test on same city
%% Run experiment
if create_binary_mask
    for ii=1:nCountries
        disp([' Creating Binary Mask... for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        for jj=1:length(cities)
            disp([' Creating Binary Mask... for ' cities{jj}])
            spectiff_to_mat(data.countries{ii}.name,cities{jj},source,type,server);
        end
    end
end
if train_model
    for ii=1:nCountries
        disp([' Training model... for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        for jj=1:length(cities)
             ccf= inform2spec(data.countries{ii}.name,source,cities{jj},type,...
                 method,split,data.countries{ii}.multiclass{jj},...
                 data.countries{ii}.classmaps{jj},data.countries{ii}.classremove{jj},...
                 ntrees,server, predict);
             data.countries{ii}.ccf{jj} = ccf; 
       end
    end
end
 if test_model
      for ii=1:nCountries
        disp(['Testing model...for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        for jj=1:length(cities)
            for kk=1:length(data.countries{ii}.testcities)
                predictinf2spec(data.countries{ii}.ccf{jj},data.countries{ii}.name,source,cities{jj},data.countries{ii}.testcities{kk},type,method,data.countries{ii}.multiclass{jj}, data.countries{ii}.classmaps{jj},data.countries{ii}.classremove{jj},ntrees,server);
            end
        end
      end
  end
if classify_image
     for ii=1:nCountries
        disp(['Classifying image ... for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        for jj=1:length(cities)
            for kk=1:length(data.countries{ii}.testcities)
                classifyimage(data.countries{ii}.ccf{jj},data.countries{ii}.name,source,cities{jj},data.countries{ii}.testcities{kk},type,method,ntrees,server)
            end
        end
     end
end
if create_image
     parfor ii=1:nCountries
        disp(['Create image...for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        for jj=1:length(cities)
            for kk=1:length(data.countries{ii}.testcities)
                plotlabelled(data.countries{ii}.name,cities{jj},data.countries{ii}.testcities{kk}, classes,cmap, lbl,ntrees,server)
            end
        end
     end
end
if calc_meanIOU
    for ii=1:nCountries
        disp(['Create image...for ' data.countries{ii}.name]);
        cities = data.countries{ii}.cities;
        parfor jj=1:length(cities)
            for kk=1:length(data.countries{ii}.testcities)
                meanIOU(data.countries{ii}.name,source,cities{jj},data.countries{ii}.testcities{kk},type,method, server)
            end
        end
     end
end
if create_all
    ground_truth =[];
    for ii=1:nCountries
         disp(['Merging all masks...currently merging ' data.countries{ii}.name]);
         ground_truth = concat_all_binary(data.countries{ii}.name,data.countries{ii}.cities,source,type,server,ground_truth);
    end
    fsave = strcat(load_inf,'All_settlements/','All/','S2/',type,'/All_ground_truth.mat');
    disp(' Saving all informal settlements binary mask ');
    save(fsave,'ground_truth');
end



