function [X,Y,bOrdinal] = loadCSVDataSet(inputLocation,bConvertCat,bNamed)
%loadCSVDataSet Loads a formatted csv to give features and classes
%
% function [X,Y,bOrdinal] = loadCSVDataSet(inputLocation)
%
% Loads a CSV to produce X, Y and bOrdinal.
%
% The CSV should be formatted as follows:
%     The last column gives the target Y (for multiple targets see below)
%     with featuers X in the other columns.  The following delimiters are 
%     supported:
%               [' ' ',' '|' ';' tab].
%     Classes can either be numerical or strings.  If the dataset contains
%     non-ordinal features then these should either be specified using
%     non-numeric symbols or by specifying a header line of the form
%               bOrdinal%1%1%0%1%...
%     where % indicates the delimiter used, 1 indicates the respective
%     feature is ordinal and 0 that it is an unordered categorical.
%     For an example of these see the processed datasets in Dataset folder
%     of the directory (soybean gives an example where bOrdinal is used).
%     If bOrdinal is not specified then variables are assumed to be ordinal
%     unless the variable has MORE than one unique non-numeric input.  If
%     there is only one non-numeric input this is assumed to correspond to
%     a missing value.  For example if a feature has values 1 2 ? 4 ? it
%     will be converted to an ordinal feature with corresponding x values
%     1 2 NaN 4 NaN, but 1 2 ? 4 a will be treated as a non-ordinal feature
%     with values '1' '2' '?' '4' 'a'.
%
%     Multiple targets can be specified by using a header line
%              n_targets%T
%     where T is the number of targets.
%
%     Header lines can be in any order, but should be before the variable
%     name line, if present.
%
% Inputs: inputLocation = String giving location of an input csv file.
%       bConvertCat = Whether to convert categorical features to numerical
%                  ones.  They will still be marked a ordinal by bOrdinal
%                  but this allows them to be passed to TreeBagger without
%                  causing issue as it doesn't allow cell inputs. Default =
%                  true.
%         bNamed = Whether the csv contains names for the variables.  If so
%                  these should come after the header lines (if
%                  they exists).  False by default
%
%
% Outpus:   X = Features, this will be a numeric array if possible and
%               otherwise a cell array.  When a cell array, cell values
%               will be numeric if all non-missing values of that feature
%               are numeric, otherwise all values in column will be
%               strings.
%           Y = Outputs returned as a a numeric column vector when
%               possible and a column vector of cells otherwise.
%    bOrdinal = Row vector of logicals indicating whether a feature is
%               ordinal (true) or an unordered categorical (false).
%
% Tom Rainforth 04/08/15

if ~exist('bConvertCat','var') || isempty(bConvertCat)
    bConvertCat = true;
end

if ~exist('bNamed','var') || isempty(bNamed)
    bNamed = false;
end

fid = fopen(inputLocation,'r');
firstLine = fgets(fid);
secondLine = fgets(fid);
bOrdSpec = [strcmpi(firstLine(1:min(8,end)),'bOrdinal'),strcmpi(secondLine(1:min(8,end)),'bOrdinal')];
bNTSpec = [strcmpi(firstLine(1:min(9,end)),'n_targets'),strcmpi(secondLine(1:min(9,end)),'n_targets')];

if bOrdSpec(1)
    ordSpecLine = firstLine;
elseif bOrdSpec(2)
    ordSpecLine = secondLine;
else
    ordSpecLine = [];
end

if bNTSpec(1)
    ntSpecLine = firstLine;
elseif bNTSpec(2)
    ntSpecLine = secondLine;
else
    ntSpecLine = [];
end

if ~isempty(ordSpecLine)
    iNum = regexp(ordSpecLine,'\d');
    bOrdinal = arrayfun(@(x) logical(str2double(ordSpecLine(x))), iNum);
else
    bOrdinal = [];
end

if ~isempty(ntSpecLine)
    n_out = str2double(ntSpecLine(regexp(ntSpecLine,'\d')));
else
    n_out = 1;
end

third_line = fgets(fid);
iDel = regexp(third_line,'[\s , \| ; \t]');
if isempty(iDel)
    error('Invalid delimiter');
end
delimiter = third_line(iDel(1));

fclose(fid);

InputTable = readtable(inputLocation,'HeaderLines',any(ordSpecLine)+any(bNTSpec),'ReadVariableNames',bNamed,'Delimiter',delimiter);

try
    Y = table2array(InputTable(:,(end-n_out+1):end));
catch
    Y = table2cell(InputTable(:,(end-n_out+1):end));
end

try
    X = table2array(InputTable(:,1:end-n_out));
    assert(~iscell(X));
    if isempty(bOrdinal)
        bOrdinal = true(1,size(X,2));
    end
catch
    X = table2cell(InputTable(:,1:end-n_out));
    bNumeric = cellfun(@isnumeric,X);
    
    if isempty(bOrdinal)
        bOrdinal = sum(~bNumeric,1)==0;
        iContainsString = find(~bOrdinal);
        dubiously_marked = false(size(bOrdinal));
    else
        % Check if something marked as ordinal is not
        apparently_numeric = sum(~bNumeric,1)==0;
        dubiously_marked = bOrdinal&~apparently_numeric;
        iContainsString = find(dubiously_marked);
    end
        
    for n=1:numel(iContainsString)
        Xnum = cellfun(@str2double,X(~bNumeric(:,iContainsString(n)),iContainsString(n)));
        bNotNum = isnan(Xnum);
        nUniqueStrings = numel(unique(X(bNotNum,iContainsString(n))));
        if nUniqueStrings<=1
            X(~bNumeric(:,iContainsString(n)),iContainsString(n)) = num2cell(Xnum);
            bOrdinal(iContainsString(n)) = true;
            dubiously_marked(iContainsString(n)) = false;
        end
    end
    
    if any(dubiously_marked)
        warning('Fields marked as ordinal contain more than one unique string, cannot treat as ordinal!');
    end
    
    if all(bOrdinal)
        X = cell2mat(X);
    end
end

if bConvertCat && iscell(X)
    for n=find(~bOrdinal)
        [~,~,iC] = unique(X(:,n));
        X(:,n) = num2cell(iC);
    end
    X = cell2mat(X);
end