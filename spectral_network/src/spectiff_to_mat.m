function spectiff_to_mat(country,city,source,type, server)
%spectrum tiff to matrix with binary classes; 
    if ~exist('server','var')
        server = '';
    end
    
        
    base = strcat(server,'Training_sets_and_ground_truth/informal_classification/');
    ext = '.tif';

    fgt = strcat(base,country,'/',city,'/',source,'/',type,'/',city,'_ground_truth',ext);
    fimage= strcat(base,country,'/',city,'/',source,'/',type,'/',city,ext);
    informal_mask = double(imread(fgt));
    base_image = double(imread(fimage));
    [row1,col1] = find(informal_mask(:,:,1));
    [row0,col0] = find(informal_mask(:,:,1)==0);

    informal_spec = zeros(size(row1,1),10);
    % can't append ones to the end if we want to take advantage
    % of parfor. 
    class_inf = ones(size(row1,1),1);
    envri_spec = zeros(size(row0,1),10);
    class_env = zeros(size(row0,1),1);

    parfor ii=1:size(row1,1)
        informal_spec(ii,:) = permute(base_image(row1(ii),col1(ii),:),[1,3,2]);
    end
    informal_spec = [informal_spec class_inf];
    parfor ii=1:size(row0,1)
        envri_spec(ii,:) = permute(base_image(row0(ii),col0(ii),:),[1,3,2]);
    end
    envri_spec = [envri_spec class_env];

    ground_truth = vertcat(informal_spec,envri_spec);
    fsave= strcat(base,country,'/',city,'/',source,'/',type,'/',city,'_ground_truth.mat');
    save(fsave, 'ground_truth','-v7.3','-nocompression');


end

