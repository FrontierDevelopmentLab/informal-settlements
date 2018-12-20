classdef optionsClassCCF
    % Options class for individual CCTs and CCF.  Please examine file for
    % complete options.  The following contructors are provided, for more
    % information on each see respective doc string.  In general one should
    % look to use either defaultOptions or defaultOptionsReg depending on
    % whether classification or regression is required.  In the case were
    % an out-of-bag-error estimate is required, one can use 
    % defaultOptionsCCFBag instead, but this will reduced performance.
    %  Constructors:
    %   defaultOptions
    %   defaultOptionsReg
    %   defaultOptionsForSingleCCTUsage
    %   defaultOptionsRegForSingleCCTUsage
    %   defaultOptionsCCFBag
    %   defaultOptionsRegCCFBag
    %   defaultOptionsRF
    %   defaultOptionsRFReg
    %   defaultOptionsRandomRotation
    %   defaultOptionsRandomRotationReg
    %   defaultOptionsRandomRotationBag
    %   defaultOptionsRandomRotationBagReg
    %   defaultOptionsRotationForest
    %   defaultOptionsRotationForestReg
    %
    % For more advanced usage, the following properties can also be set.
    % We split these up depending on how likely you are to want to set them
    % directly.  Option shown in brackets is the default.
    %
    % Common parameters that can be tuned if required
    %
    %   lambda = ('log') | 'sqrt' | +ve integer  
    %            Number of features to subsample at each node or 'log' 
    %            for ceil(log2(D)+1)) or 'sqrt' for ceil(sqrt(D))
    %   splitCriterion = ('info') | 'gini' | 'mse'
    %            Split criterion / impurity measure to use.  Default is
    %            'info' for classification with is entropy impurity /
    %            information split criterion.  Default is 'mse', i.e. mean
    %            squared error for classicifation
    %   minPointsLeaf = (1) | +ve integer
    %            Minimum number of points allowed a leaf node for split to
    %            be permitted.  Default is 1 for classification and 3 for
    %            regression.  Setting larger can guard against overfitting,
    %            particularly in the regression case.
    %   propTrain = (1) | (0,1]
    %            Proportion of the data to train each tree on.  Generally
    %            wants to be set to 1, but for large datasets it may be
    %            possible to only use a subset of the data for training
    %            each tree.
    %
    % Common computational options  
    %
    %   bUseParallel = (true) | false 
    %            Whether to use the parallel toolbox for training / testing
    %   bCalcTimingStats = (true) | false
    %            Option for calculating timing stats.  Stores total wall 
    %            time, tree training times, number of nodes for each tree, 
    %            and if appropriate the tree test times.  When running in 
    %            parallel, the tree timings will sum up to a larger value 
    %            then the overall wall time.
    %   
    % Options dictating type of estimation done
    % 
    %   bSepPred = (false) | true
    %            Whether to predict each class seperately as a multilabel
    %            classification problem (true) or treat classes within the
    %            same output as mutually exclusive (false). If set to 
    %            false, and multiple outputs or provided, then the maximal
    %            left-to-right grouping of outputs is found such that each
    %            output always has one class.  If the number of present 
    %            classes varies between datapoints, options will be 
    %            overwritten to true.
    %   taskWeights = 'even' | vector of +ve reals
    %            Weights to apply to each output task in calculating the 
    %            gain. Should either be 'even' dictating all outputs are
    %            given the same weight, or a vector of relative weights 
    %            each to the number of outputs.
    %
    % Options dictating type of estimator to train (e.g. CCF or CCF-Bag)
    %
    %   bProjBoot = ('default') | true | false  
    %            Whether to use projection bootstrapping.  If set to 
    %            'default', then true unless lambda=D, i.e. we all features
    %            at each node.  In this case we resort to bagging instead
    %            of projection bootstrapping as explained in the paper.
    %   bBagTrees = ('default') | true | false  
    %            Whether to use Breiman's bagging by training each tree on 
    %            a bootstrap sample  Default is true unless lambda=D
    %   projections = struct('CCA',true) | see text below
    %            Projections to consider for splitting at each node.  Must 
    %            be a structure with the possible fields 'CCA', 'PCA', 
    %            'CCAclasswise' (i.e. do CCA on each output / class 
    %            seperately), 'Original' and 'Random' (i.e. random 
    %            rotation).  Each field should be either be logical or a 
    %            number between 0 and 1.  When a field is true that
    %            projection is always performed If values are reals, then a
    %            projection is sampled with proportional probability from 
    %            those fields.  For example 
    %                struct('CCA',true,'PCA',0.3,','Random',0.7)
    %            will always do a CCA and one of PCA and Random with 
    %            respective probabilities 0.3 and 0.7.  Note this sampling 
    %            is done independently at each projection calculation.
    %   treeRotation = ('none') | 'pca' | 'random' | 'rotationForest'
    %            Pre-rotation to be applied to each tree seperately before
    %            rotating.
    %
    % Options that may want to be set if using algorithms building on CCFs
    % (mostly rotation forest stuff)
    %
    %   RotForM = 3 | +ve integer; 
    %             Size of feature subsets taken for each rotation.  Default
    %             as per WEKA and rotation forest paper
    %   RotForpS = 0.5; 
    %             Proportion of points to subsample for calculating each 
    %             PCA projection.  Default as per WEKA but not rotation 
    %             forest paper (which takes RotForpS = 0.75)
    %   RotForpClassLeaveOut = 0.5; 
    %             Proportion of classes to randomly eliminate for each
    %             PCA projection.  Though this is mentioned in the paper,
    %             there is no actually strategy or parameter values given
    %             and subsequent work / WEKA implementation does not
    %             mention it.  This is a little bizarre as the parameter 
    %             seems paramount to the performance of rotation forest 
    %             and the performance cannot be matched if set to 0 as some
    %             implementations do.
    %
    % Numerical stability options.  May wish to set these in particular
    % scenarios but usually want leaving alone.
    % 
    %   epsilonCCA = (1e-4) | +ve real 
    %           Tolerance parameter for rank reduction during the CCA.  It 
    %           can be desirable to lower if the data has extreme 
    %           correlation, in which this finite value could eliminate the
    %           true signal.
    %   mseErrorTolerance = 1e-6 | +ve real
    %           When doing regression with mse splits, the node is made 
    %           into a leaf if the mse (i.e. variance) of the data is less 
    %           than this tolerance times the mse of the full data set.
    %           This is set to match other packages and provides a degree
    %           of regularization.  If aiming for very high accuracies in
    %           the regressor, this may want lowering.  Usually this is
    %           much less than the accuracy of the regressor itself though
    %           so beneficial to have it be non-zero.
    %  maxDepthSplit = 'stack' | +ve integer
    %           Maximum depth of a node when splitting is still allowed.
    %           When set to 'stack' this is set to the maximum value that
    %           prevents crashes (usually ~500 which should never really be
    %           reached in sensible scenarios).  Though in theory this is a
    %           parameter of the algorithm one might want to change for
    %           non-computational reasons, it is rarely beneficial compared
    %           to say changing minPointsLeaf.  Might want to set it to
    %           improve predictive speed though.
    %  XVariationTol = 1e-10 | +ve integer
    %           Points closer than this tolerance (after scaling the data
    %           to unit standard deviation) are considered the same the
    %           avoid splitting on numerical error.  Rare would want to
    %           change.
    %
    % Other properties:
    %   There are also some other properties though these do not generally
    %   want to be set, either because they are very minor choices, or
    %   because they pertain to possible extensions to the algorithm (e.g.
    %   using kernel CCA).  There are also some internally set properties.
    %   Information for both of these is provided in comments of this file.
    % 
    % 23/07/17
    
    
    properties
        %% Properties that can be set is desired
        
        lambda = 'log';
        splitCriterion = 'info'; %TODO add alternatives for regression (e.g. curvature)
        minPointsLeaf = 1;
        bUseParallel = true;
        bCalcTimingStats = true;
        bSepPred = false;
        taskWeights = 'even';
        bProjBoot = 'default';
        bBagTrees = 'default';
        projections = struct('CCA',true);
        treeRotation = 'none';
        propTrain = 1;
        
        %% Properties relating to numerical stability
        % Might want to set in certain scenarios explained in the doc
        % string
        
        epsilonCCA = 1e-4;
        mseErrorTolerance = 1e-6;
        maxDepthSplit = 'stack';
        XVariationTol = 1e-10;
        
        %% Properties that may want to be set if using other supported algs
        
        % Rotation forest options
        RotForM = 3;
        RotForpS = 0.75;
        RotForpClassLeaveOut = 0.5;
        
        %% Properties that can be set but should generally be avoided
        
        minPointsForSplit = 2; % Minimum points for parent node, 
                               % Default = 6 for regression
        
        
        % When multiple projection vectors can give equivalent split
        % criterion scores, one can either choose which to use randomly
        % ('rand') or take the first ('first') on the basis that the
        % components are in decreasing order of correlation for CCA.  If
        % using more than just CCA for projections, using 'first' also
        % allows preferential treatment to certain projections
        dirIfEqual = 'first';
        
        % In the scenario where the projection bootstrap makes the local
        % data pure or have no X variation, the algorithm can either set
        % the node to be a leaf or resort to using the original data for
        % the CCA
        bContinueProjBootDegenerate = true;
        
        % Method for combining multiple gain metrics in multi-output
        % tasks.  Valid options are 'mean' (default) - average of the
        % gains which for all the considered metrics is equal to the
        % joint gain, or the 'max' gain on any of the tasks.
        multiTaskGainCombination = 'mean';
        
        % Method for dealing with missing values.  Should be either
        % 'random' which sets the missing value randomly to one of the
        % other values for each tree seperately or 'mean' which sets
        % missing values to the mean.
        missingValuesMethod = 'random';
        
        % If true, doing regression with multiple outputs and doing CCA
        % projections, then the mse of the output components is calculated
        % instead of the original values
        bUseOutputComponentsMSE = false;
        
        % Options that allow nonlinear features to be included in the CCA
        % in accordance with Lopez-Paz's randomized kernel cca.  Initial
        % results suggest that this is not a useful expansion of the model
        bRCCA = false;
        rccaLengthScale = 0.1;
        rccaNFeatures = 50;
        rccaRegLambda = 1e-3;
        rccaIncludeOriginal = false;
        
        
        %% Properties that are not set but used as a place to store info
        
        classNames     % Used to convert the predictors back to their class names
        
        org_muY        % Used in regression for converting the Z scores back
        org_stdY       % to original values when creating the leaf model
        
        mseTotal       % Used as a baseline in regression with
        % mseErrorTolerance for early termination when
        % there is negiligable variation in the outputs.
        % Terminates if mse<mseTotal*mseErrorTolerance
        
        task_ids       % Gives groupings of outputs in classification.  Is
        % an array where each value is the start of a new
        % output block.
        
    end
    
    %%
    
    methods
        
        function obj = updateForD(obj,D)
            % Updates the options for a particular D to set lambda and
            % decide whether to do projection bootstrapping (by default
            % this is done unless lambda==D).
            
            if strcmpi(obj.lambda,'sqrt')
                obj.lambda = ceil(sqrt(D));
            elseif strcmpi(obj.lambda,'log')
                if D==3
                    obj.lambda = 2;
                else
                    obj.lambda = ceil(log2(D)+1);
                end
            elseif strcmpi(obj.lambda,'all')
                obj.lambda = D;
            elseif ~isnumeric(obj.lambda)
                error('Invalid option set for lambda');
            end
            
            if strcmpi(obj.bProjBoot,'default')
                if D<=obj.lambda
                    obj.bProjBoot = false;
                else
                    obj.bProjBoot = true;
                end
            end
            
            if strcmpi(obj.bBagTrees,'default')
                if D<=obj.lambda
                    obj.bBagTrees = true;
                else
                    obj.bBagTrees = false;
                end
            end
            
        end
    end
    
    methods (Static)
        
        function obj = defaultOptions
            % Contructs default options for classification
            obj = optionsClassCCF;
        end
        
        function obj = defaultOptionsReg
            % Constructs default options for regression
            
            obj = optionsClassCCF;
            obj.splitCriterion = 'mse';
            obj.minPointsForSplit = 6;
            obj.minPointsLeaf = 3;
        end
        
        function obj = defaultOptionsForSingleCCTUsage
            % Allows easy calling of a default options set for when growing
            % a single CCT tree for individual use rather than as part of a
            % forest.
            
            obj = optionsClassCCF;
            obj.bProjBoot = false;
            obj.lambda = 'all';
            obj.bBagTrees = false;
        end
        
        function obj = defaultOptionsRegForSingleCCTUsage
            % Allows easy calling of a default options set for when growing
            % a single CCT tree for individual use rather than as part of a
            % forest.
            
            obj = optionsClassCCF.defaultOptionsReg;
            obj.bProjBoot = false;
            obj.lambda = 'all';
            obj.bBagTrees = false;
        end
        
        
        function obj = defaultOptionsCCFBag
            % Automatically sets up options for running CCF-Bag which uses
            % bagging instead of projection bootstrapping (please see the
            % paper for further details).  This options should be used if
            % the oob error is needed (e.g. for hyperparameter
            % optimization).
            
            obj = optionsClassCCF;
            obj.bProjBoot = false;
            obj.bBagTrees = true;
        end
        
        function obj = defaultOptionsRegCCFBag
            % Constructs default options for regression and CCFBag
            
            obj = optionsClassCCF.defaultOptionsReg;
            obj.bProjBoot = false;
            obj.bBagTrees = true;
        end
        

        
        function obj = defaultOptionsRF
            % Sets up options for running the package as a random forest
            % package.  This is not recommended as general usage (as more
            % efficient implementation exist), but is a useful debugger to
            % check that the differences come from the CCF extension rather
            % than unexpected options variations.  It is also used for
            % like-for-like timing comparisons
            
            obj = optionsClassCCF;
            obj.bBagTrees = true;
            obj.bProjBoot = false;
            obj.projections = struct('Original',true);
        end
        
        function obj = defaultOptionsRFReg
            % As optionsClassCCF.defaultOptionsRF but for regression
            
            obj = optionsClassCCF.defaultOptionsRF;
            obj.splitCriterion = 'mse';
            obj.minPointsForSplit = 6;
            obj.minPointsLeaf = 3;
        end
        
        function obj = defaultOptionsRandomRotation
            % Allows for use of random rotation instead of CCA.
            % Corresponds to RRF from the paper.
            
            obj = optionsClassCCF;
            obj.projections = struct('Random',true);
        end
        
        function obj = defaultOptionsRandomRotationReg
            % As defaultOptionsRandomRotation but for regression
            
            obj = optionsClassCCF.defaultOptionsRandomRotation;
            obj.splitCriterion = 'mse';
            obj.minPointsForSplit = 6;
            obj.minPointsLeaf = 3;
        end
        
        function obj = defaultOptionsRandomRotationBag
            % Allows for use of random rotation instead of CCA and bagging
            % instead of projection bootrapping.  Corresponds to RRF-Bag
            % from the paper
            
            obj = optionsClassCCF.defaultOptionsRandomRotation;
            obj.bProjBoot = false;
            obj.bBagTrees = true;
        end
        
        function obj = defaultOptionsRandomRotationBagReg
            % As defaultOptionsRandomRotationBag but for regression
            
            obj = optionsClassCCF.defaultOptionsRandomRotationReg;
            obj.bProjBoot = false;
            obj.bBagTrees = true;
        end
        
        function obj = defaultOptionsRotationForest
            % Use the package for training and predicting from a rotation
            % forest.
            
            warning(['Results often vary significantly compared to Weka (for '...
                'both better and worse) for rotation forest, details that '...
                'are missing from the paper appear to be important.']);
            obj = optionsClassCCF;
            obj.bProjBoot = false;
            obj.projections = struct('Original',true);
            obj.treeRotation = 'rotationForest';
            obj.lambda = 'all';
            obj.bBagTrees = false;
        end
        
        function obj = defaultOptionsRotationForestReg
            % Use the package for training and predicting from a rotation
            % forest for regression
            
            obj = optionsClassCCF.defaultOptionsRotationForest;
            obj.splitCriterion = 'mse';
            obj.minPointsForSplit = 6;
            obj.minPointsLeaf = 3;
        end
    end
    
end