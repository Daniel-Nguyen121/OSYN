from yacs.config import CfgNode as CN
_C = CN()

#Classifiers
_C.CLS = CN()
_C.CLS.RANDOM_STATE = 42       #Random state to generate the random permutations for shuffling the data. Used in SVC, GB, MLP, LR, DT, RF.
_C.CLS.VERBOSE = 0       #Controls the verbosity of the tree building process. Used in SVC, RF, GB, MLP, LR.
_C.CLS.N_JOBS = None      #Number of jobs to run in parallel. Used in KNN, RF, LR.
_C.CLS.TOL = 1e-4         #Tolerance for stopping criteria. Used in SVC, LDA, GB, MLP, LR.
_C.CLS.CLASS_WEIGHT = None #Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. Used in SVC, DT, RF, LR.
_C.CLS.WARM_START = False #Whether to reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Used in RF, MLP, LR, GB.

#-----------------------------------------------#
#####   KNN    #####
#-----------------------------------------------#
_C.CLS.KNN = CN()
_C.CLS.KNN.N_NEIGHBORS = 5          #Number of neighbors to use by default for kneighbors queries.
_C.CLS.KNN.WEIGHTS = 'uniform'      #Weight function used in prediction. 'uniform' or 'distance'.
_C.CLS.KNN.P = 2                    #p=1: Manhattan distance, p=2: Euclidean distance
_C.CLS.KNN.METRIC = 'minkowski'     #'minkowski', 'manhattan', 'euclidean', 'chebyshev', 'hamming', 'canberra', 'braycurtis', 'mahalanobis', 'cosine', 'correlation', 'dice', 'jaccard', 'hamming', 'jensenshannon', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule' # Metric to use for distance computation.
_C.CLS.KNN.METRIC_PARAMS = None     #Additional parameters to pass to the metric function.
_C.CLS.KNN.ALGORITHM = 'auto'       #'auto', 'ball_tree', 'kd_tree', 'brute'
_C.CLS.KNN.LEAF_SIZE = 30           #Leaf size passed to BallTree or KDTree

#-----------------------------------------------#
#####   Linear SVC    #####
#-----------------------------------------------#
_C.CLS.SVC = CN()
_C.CLS.SVC.PENALTY = 'l2'             #Penalty parameter of the error term. 'l1' or 'l2'
_C.CLS.SVC.LOSS = 'squared_hinge'     #Loss function, 'hinge', 'squared_hinge'
_C.CLS.SVC.DUAL = 'auto'              #Whether to use the dual formulation.
_C.CLS.SVC.C = 1.0                    #Penalty parameter of the error term.
_C.CLS.SVC.MULTI_CLASS = 'ovr'        #'ovr' or 'ovo'
_C.CLS.SVC.FIT_INTERCEPT = True       #Whether to calculate the intercept for this model.
_C.CLS.SVC.INTERCEPT_SCALING = 1.0    #Scale the intercept for this model.
_C.CLS.SVC.MAX_ITER = 1000            #Maximum number of iterations taken for the solvers to converge.

#-----------------------------------------------#
#####   Decision Tree    #####
#-----------------------------------------------#
_C.CLS.DT = CN()
_C.CLS.DT.CRITERION = 'gini'        #The function to measure the quality of a split.
_C.CLS.DT.SPLITTER = 'best'         #The strategy used to choose the split at each node.
_C.CLS.DT.MAX_DEPTH = None           #The maximum depth of the tree.
_C.CLS.DT.MIN_SAMPLES_SPLIT = 2      #The minimum number of samples required to split an internal node.
_C.CLS.DT.MIN_SAMPLES_LEAF = 1       #The minimum number of samples required to be at a leaf node.
_C.CLS.DT.MIN_WEIGHT_FRACTION_LEAF = 0.00 #The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
_C.CLS.DT.MAX_FEATURES = None        #The number of features to consider when looking for the best split.
_C.CLS.DT.MAX_LEAF_NODES = None      #The maximum number of leaf nodes.
_C.CLS.DT.MIN_IMPURITY_DECREASE = 0.0 #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
_C.CLS.DT.CCP_ALPHA = 0.0            #Complexity parameter used for Minimal Cost-Complexity Pruning.
_C.CLS.DT.MONOTONIC_CST = None      #Whether to enforce monotonic constraints on the weights.

#-----------------------------------------------#
#####   Random Forest    #####
#-----------------------------------------------#
_C.CLS.RF = CN()
_C.CLS.RF.N_ESTIMATORS = 10        #The number of trees in the forest.
_C.CLS.RF.CRITERION = 'gini'        #The function to measure the quality of a split: {'gini', 'entropy', 'log_loss'}.
_C.CLS.RF.MAX_DEPTH = None           #The maximum depth of the tree.
_C.CLS.RF.MIN_SAMPLES_SPLIT = 2      #The minimum number of samples required to split an internal node.
_C.CLS.RF.MIN_SAMPLES_LEAF = 1       #The minimum number of samples required to be at a leaf node.
_C.CLS.RF.MIN_WEIGHT_FRACTION_LEAF = 0.0 #The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
_C.CLS.RF.MAX_FEATURES = 1      #The number of features to consider when looking for the best split: {'sqrt', 'log2', None or int}.
_C.CLS.RF.MAX_LEAF_NODES = None      #The maximum number of leaf nodes.
_C.CLS.RF.MIN_IMPURITY_DECREASE = 0.0 #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
_C.CLS.RF.BOOTSTRAP = True           #Whether bootstrap samples are used when building trees.
_C.CLS.RF.OOB_SCORE = False          #Whether to use out-of-bag samples to estimate the generalization score.
_C.CLS.RF.CCP_ALPHA = 0.0             #Complexity parameter used for Minimal Cost-Complexity Pruning.
_C.CLS.RF.MAX_SAMPLES = None          #The maximum number of samples to draw from X to train each base estimator.
_C.CLS.RF.MONOTONIC_CST = None        #Whether to enforce monotonic constraints on the weights.

#-----------------------------------------------#
#####           Multilayer Perceptron       #####
#-----------------------------------------------#
_C.CLS.MLP = CN()
_C.CLS.MLP.HIDDEN_LAYER_SIZES = (100,)  #The number of neurons in each hidden layer.
_C.CLS.MLP.ACTIVATION = 'relu'          #The activation function to use in the hidden layers: {'relu', 'logistic', 'tanh', 'identity'}.
_C.CLS.MLP.SOLVER = 'adam'              #The optimizer to use: {'lbfgs', 'sgd', 'adam'}.
_C.CLS.MLP.ALPHA = 0.0001                #L2 penalty (regularization term) parameter.
_C.CLS.MLP.BATCH_SIZE = 'auto'          #The size of the mini-batches for stochastic optimizers, batch_size=min(200, n_samples).
_C.CLS.MLP.LEARNING_RATE = 'constant'    #The initial learning rate for the optimizer: {'constant', 'invscaling', 'adaptive' - only used when solver='sgd'}.
_C.CLS.MLP.LEARNING_RATE_INIT = 0.001    #The initial learning rate for the optimizer, only used when solver='sgd' or 'adam'.
_C.CLS.MLP.POWER_T = 0.5                 #The power to which the learning rate is raised when the learning rate schedule is 'power_t', only used when solver='sgd'.
_C.CLS.MLP.MAX_ITER = 1000               #Increased from 200 to 1000 to allow more iterations for convergence
_C.CLS.MLP.SHUFFLE = True                #Whether to shuffle the data before each iteration. Only used when solver='sgd' or 'adam'.
_C.CLS.MLP.MOMENTUM = 0.9                #The momentum for gradient descent updates.
_C.CLS.MLP.NESTEROVS_MOMENTUM = True     #Whether to use Nesterov's momentum.
_C.CLS.MLP.EARLY_STOPPING = True         #Changed to True to stop early if validation score doesn't improve
_C.CLS.MLP.VALIDATION_FRACTION = 0.1     #The fraction of the training data to use for validation.
_C.CLS.MLP.BETA_1 = 0.9                  #The beta1 hyperparameter for Adam.
_C.CLS.MLP.BETA_2 = 0.999                #The beta2 hyperparameter for Adam.
_C.CLS.MLP.EPSILON = 1e-8                #The epsilon hyperparameter for Adam.
_C.CLS.MLP.N_ITER_NO_CHANGE = 10         #The number of iterations with no improvement to wait before stopping. Only used when solver='sgd' or 'adam'.
_C.CLS.MLP.MAX_FUN = 15000                #The maximum number of function evaluations. Only used when solver='lbfgs'.

#-----------------------------------------------#
#####   Logistic Regression    #####
#-----------------------------------------------#
_C.CLS.LR = CN()
_C.CLS.LR.PENALTY='l2'                #The penalty (aka regularization term) to use: {'l1', 'l2', 'elasticnet', 'none'}.
_C.CLS.LR.DUAL=False                  #Whether to use the dual formulation.
_C.CLS.LR.C=1.0                        #Penalty parameter of the error term.
_C.CLS.LR.FIT_INTERCEPT=True           #Whether to calculate the intercept for this model.
_C.CLS.LR.INTERCEPT_SCALING=1          #Scale the intercept for this model.
_C.CLS.LR.SOLVER='lbfgs'               #The solver to use: {'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'}.
_C.CLS.LR.MAX_ITER=1000               #Increased from 100 to 1000 for better convergence
_C.CLS.LR.MULTI_CLASS='multinomial'    #Changed from 'auto' to 'multinomial' to fix deprecation warning
_C.CLS.LR.L1_RATIO=None

#-----------------------------------------------#
#####   Gradient Boosting    #####
#-----------------------------------------------#
_C.CLS.GB = CN()
_C.CLS.GB.LOSS='log_loss'                  #The loss function to use: {'log_loss', 'exponential'}.
_C.CLS.GB.LEARNING_RATE=0.1          #Learning rate shrinks the contribution of each tree by learning_rate.
_C.CLS.GB.N_ESTIMATORS=100           #The number of trees in the forest.
_C.CLS.GB.SUBSAMPLE=1.0              #The fraction of samples to draw from X to train each tree.
_C.CLS.GB.CRITERION='friedman_mse'   #The criterion to measure the quality of a split: {'friedman_mse', 'squared_error'}.
_C.CLS.GB.MIN_SAMPLES_SPLIT=2         #The minimum number of samples required to split an internal node.
_C.CLS.GB.MIN_SAMPLES_LEAF=1          #The minimum number of samples required to be at a leaf node.
_C.CLS.GB.MIN_WEIGHT_FRACTION_LEAF=0.0 #The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
_C.CLS.GB.MAX_DEPTH=3                 #The maximum depth of the individual regression estimators.
_C.CLS.GB.MIN_IMPURITY_DECREASE=0.0   #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
_C.CLS.GB.INIT=None                   #The initial estimator to use when warm_start=True: {'zero', None}.
_C.CLS.GB.MAX_FEATURES=None           #The number of features to consider when looking for the best split: {'sqrt', 'log2', None or int}.
_C.CLS.GB.MAX_LEAF_NODES=None         #The maximum number of leaf nodes.
_C.CLS.GB.VALIDATION_FRACTION=0.1     #The fraction of the training data to use for validation.
_C.CLS.GB.N_ITER_NO_CHANGE=None       #The number of iterations with no improvement to wait before stopping.
_C.CLS.GB.CCP_ALPHA=0.0                #Complexity parameter used for Minimal Cost-Complexity Pruning.

#-----------------------------------------------#
#####   Linear Discriminant Analysis    #####
#-----------------------------------------------#
_C.CLS.LDA = CN()
_C.CLS.LDA.SOLVER='svd'                 #The solver to use for computing the covariance matrix: {'svd', 'lsqr', 'eigen'}.
_C.CLS.LDA.SHRINKAGE=None               #The amount of shrinkage to apply: {'auto', None, float}.
_C.CLS.LDA.PRIORS=None                  #The prior probabilities of the classes.
_C.CLS.LDA.N_COMPONENTS=None            #The number of components to keep.
_C.CLS.LDA.STORE_COVARIANCE=False       #Whether to store the covariance matrix.
_C.CLS.LDA.COVARIANCE_ESTIMATOR=None    #The estimator to use for computing the covariance matrix.

def get_hyper_params_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone() 