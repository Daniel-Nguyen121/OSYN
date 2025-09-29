from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classifier_config import get_hyper_params_defaults
from config import get_cfg_defaults

#Read configuration
cfg = get_cfg_defaults()
hp = get_hyper_params_defaults()

def get_knn_classifier(n_neighbors=hp.CLS.KNN.N_NEIGHBORS, \
                      weights=hp.CLS.KNN.WEIGHTS, \
                      algorithm=hp.CLS.KNN.ALGORITHM, \
                      leaf_size=hp.CLS.KNN.LEAF_SIZE, \
                      p=hp.CLS.KNN.P, \
                      metric=hp.CLS.KNN.METRIC, \
                      metric_params=hp.CLS.KNN.METRIC_PARAMS, \
                      n_jobs=hp.CLS.N_JOBS):

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, \
                              weights=weights, \
                              algorithm=algorithm, \
                              leaf_size=leaf_size, \
                              p=p, \
                              metric=metric, \
                              metric_params=metric_params, \
                              n_jobs=n_jobs)
    return clf

def get_svm_classifier(penalty=hp.CLS.SVC.PENALTY, \
                       loss=hp.CLS.SVC.LOSS, \
                       dual=hp.CLS.SVC.DUAL, \
                       tol=hp.CLS.TOL, \
                       C=hp.CLS.SVC.C, \
                       multi_class=hp.CLS.SVC.MULTI_CLASS, \
                       fit_intercept=hp.CLS.SVC.FIT_INTERCEPT, \
                       intercept_scaling=hp.CLS.SVC.INTERCEPT_SCALING, \
                       class_weight=hp.CLS.CLASS_WEIGHT, \
                       verbose=hp.CLS.VERBOSE, \
                       random_state=hp.CLS.RANDOM_STATE, \
                       max_iter=hp.CLS.SVC.MAX_ITER):
    clf = LinearSVC(penalty=penalty, \
                    loss=loss, \
                    dual=dual, \
                    tol=tol, \
                    C=C, \
                    multi_class=multi_class, \
                    fit_intercept=fit_intercept, \
                    intercept_scaling=intercept_scaling, \
                    class_weight=class_weight, \
                    verbose=verbose, \
                    random_state=random_state, \
                    max_iter=max_iter)
    return clf

def get_dt_classifier(criterion=hp.CLS.DT.CRITERION, \
                      splitter=hp.CLS.DT.SPLITTER, \
                      max_depth=hp.CLS.DT.MAX_DEPTH, \
                      min_samples_split=hp.CLS.DT.MIN_SAMPLES_SPLIT, \
                      min_samples_leaf=hp.CLS.DT.MIN_SAMPLES_LEAF, \
                      min_weight_fraction_leaf=hp.CLS.DT.MIN_WEIGHT_FRACTION_LEAF, \
                      max_features=hp.CLS.DT.MAX_FEATURES, \
                      random_state=hp.CLS.RANDOM_STATE, \
                      max_leaf_nodes=hp.CLS.DT.MAX_LEAF_NODES, \
                      min_impurity_decrease=hp.CLS.DT.MIN_IMPURITY_DECREASE, \
                      class_weight=hp.CLS.CLASS_WEIGHT, \
                      ccp_alpha=hp.CLS.DT.CCP_ALPHA, \
                      monotonic_cst=hp.CLS.DT.MONOTONIC_CST):
    clf = DecisionTreeClassifier(criterion=criterion, \
                                 splitter=splitter, \
                                 max_depth=max_depth, \
                                 min_samples_split=min_samples_split, \
                                 min_samples_leaf=min_samples_leaf, \
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, \
                                 max_features=max_features, \
                                 random_state=random_state, \
                                 max_leaf_nodes=max_leaf_nodes, \
                                 min_impurity_decrease=min_impurity_decrease, \
                                 class_weight=class_weight, \
                                 ccp_alpha=ccp_alpha, \
                                 monotonic_cst=monotonic_cst)
    return clf

def get_rf_classifier(n_estimators=hp.CLS.RF.N_ESTIMATORS, \
                      criterion=hp.CLS.RF.CRITERION, \
                      max_depth=hp.CLS.RF.MAX_DEPTH, \
                      min_samples_split=hp.CLS.RF.MIN_SAMPLES_SPLIT, \
                      min_samples_leaf=hp.CLS.RF.MIN_SAMPLES_LEAF, \
                      min_weight_fraction_leaf=hp.CLS.RF.MIN_WEIGHT_FRACTION_LEAF, \
                      max_features=hp.CLS.RF.MAX_FEATURES, \
                      max_leaf_nodes=hp.CLS.RF.MAX_LEAF_NODES, \
                      min_impurity_decrease=hp.CLS.RF.MIN_IMPURITY_DECREASE, \
                      bootstrap=hp.CLS.RF.BOOTSTRAP, \
                      oob_score=hp.CLS.RF.OOB_SCORE, \
                      n_jobs=hp.CLS.N_JOBS, \
                      random_state=hp.CLS.RANDOM_STATE, \
                      verbose=hp.CLS.VERBOSE, \
                      warm_start=hp.CLS.WARM_START, \
                      class_weight=hp.CLS.CLASS_WEIGHT, \
                      ccp_alpha=hp.CLS.RF.CCP_ALPHA, \
                      max_samples=hp.CLS.RF.MAX_SAMPLES, \
                      monotonic_cst=hp.CLS.RF.MONOTONIC_CST):
    clf = RandomForestClassifier(n_estimators=n_estimators, \
                                 criterion=criterion, \
                                 max_depth=max_depth, \
                                 min_samples_split=min_samples_split, \
                                 min_samples_leaf=min_samples_leaf, \
                                 min_weight_fraction_leaf=min_weight_fraction_leaf, \
                                 max_features=max_features, \
                                 max_leaf_nodes=max_leaf_nodes, \
                                 min_impurity_decrease=min_impurity_decrease, \
                                 bootstrap=bootstrap, \
                                 oob_score=oob_score, \
                                 n_jobs=n_jobs, \
                                 random_state=random_state, \
                                 verbose=verbose, \
                                 warm_start=warm_start, \
                                 class_weight=class_weight, \
                                 ccp_alpha=ccp_alpha, \
                                 max_samples=max_samples, \
                                 monotonic_cst=monotonic_cst)
    return clf

def get_mlp_classifier(hidden_layer_sizes=hp.CLS.MLP.HIDDEN_LAYER_SIZES, \
                       activation=hp.CLS.MLP.ACTIVATION, \
                       solver=hp.CLS.MLP.SOLVER, \
                       alpha=hp.CLS.MLP.ALPHA, \
                       batch_size=hp.CLS.MLP.BATCH_SIZE, \
                       learning_rate=hp.CLS.MLP.LEARNING_RATE, \
                       learning_rate_init=hp.CLS.MLP.LEARNING_RATE_INIT, \
                       power_t=hp.CLS.MLP.POWER_T, \
                       max_iter=hp.CLS.MLP.MAX_ITER, \
                       shuffle=hp.CLS.MLP.SHUFFLE, \
                       random_state=hp.CLS.RANDOM_STATE, \
                       tol=hp.CLS.TOL, \
                       verbose=hp.CLS.VERBOSE, \
                       warm_start=hp.CLS.WARM_START, \
                       momentum=hp.CLS.MLP.MOMENTUM, \
                       nesterovs_momentum=hp.CLS.MLP.NESTEROVS_MOMENTUM, \
                       early_stopping=hp.CLS.MLP.EARLY_STOPPING, \
                       validation_fraction=hp.CLS.MLP.VALIDATION_FRACTION, \
                       beta_1=hp.CLS.MLP.BETA_1, \
                       beta_2=hp.CLS.MLP.BETA_2, \
                       epsilon=hp.CLS.MLP.EPSILON, \
                       n_iter_no_change=hp.CLS.MLP.N_ITER_NO_CHANGE, \
                       max_fun=hp.CLS.MLP.MAX_FUN):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, \
                        activation=activation, \
                        solver=solver, \
                        alpha=alpha, \
                        batch_size=batch_size, \
                        learning_rate=learning_rate, \
                        learning_rate_init=learning_rate_init, \
                        power_t=power_t, \
                        max_iter=max_iter, \
                        shuffle=shuffle, \
                        random_state=random_state, \
                        tol=tol, \
                        verbose=verbose, \
                        warm_start=warm_start, \
                        momentum=momentum, \
                        nesterovs_momentum=nesterovs_momentum, \
                        early_stopping=early_stopping, \
                        validation_fraction=validation_fraction, \
                        beta_1=beta_1, \
                        beta_2=beta_2, \
                        epsilon=epsilon, \
                        n_iter_no_change=n_iter_no_change, \
                        max_fun=max_fun)
    return clf

def get_lr_classifier(penalty=hp.CLS.LR.PENALTY, \
                      dual=hp.CLS.LR.DUAL, \
                      tol=hp.CLS.TOL, \
                      C=hp.CLS.LR.C, \
                      fit_intercept=hp.CLS.LR.FIT_INTERCEPT, \
                      intercept_scaling=hp.CLS.LR.INTERCEPT_SCALING, \
                      class_weight=hp.CLS.CLASS_WEIGHT, \
                      random_state=hp.CLS.RANDOM_STATE, \
                      solver=hp.CLS.LR.SOLVER, \
                      max_iter=hp.CLS.LR.MAX_ITER, \
                      verbose=hp.CLS.VERBOSE, \
                      warm_start=hp.CLS.WARM_START, \
                      n_jobs=hp.CLS.N_JOBS, \
                      l1_ratio=hp.CLS.LR.L1_RATIO):
    clf = LogisticRegression(penalty=penalty, \
                              dual=dual, \
                              tol=tol, \
                              C=C, \
                              fit_intercept=fit_intercept,\
                              intercept_scaling=intercept_scaling, \
                              class_weight=class_weight, \
                              random_state=random_state, \
                              solver=solver, \
                              max_iter=max_iter, \
                              verbose=verbose, \
                              warm_start=warm_start, \
                              n_jobs=n_jobs, \
                              l1_ratio=l1_ratio)
    return clf

def get_gb_classifier(loss=hp.CLS.GB.LOSS, \
                      learning_rate=hp.CLS.GB.LEARNING_RATE, \
                      n_estimators=hp.CLS.GB.N_ESTIMATORS, \
                      subsample=hp.CLS.GB.SUBSAMPLE, \
                      criterion=hp.CLS.GB.CRITERION, \
                      min_samples_split=hp.CLS.GB.MIN_SAMPLES_SPLIT, \
                      min_samples_leaf=hp.CLS.GB.MIN_SAMPLES_LEAF, \
                      min_weight_fraction_leaf=hp.CLS.GB.MIN_WEIGHT_FRACTION_LEAF, \
                      max_depth=hp.CLS.GB.MAX_DEPTH, \
                      min_impurity_decrease=hp.CLS.GB.MIN_IMPURITY_DECREASE, \
                      init=hp.CLS.GB.INIT, \
                      random_state=hp.CLS.RANDOM_STATE, \
                      max_features=hp.CLS.GB.MAX_FEATURES, \
                      verbose=hp.CLS.VERBOSE, \
                      max_leaf_nodes=hp.CLS.GB.MAX_LEAF_NODES, \
                      warm_start=hp.CLS.WARM_START, \
                      validation_fraction=hp.CLS.GB.VALIDATION_FRACTION, \
                      n_iter_no_change=hp.CLS.GB.N_ITER_NO_CHANGE, \
                      tol=hp.CLS.TOL, \
                      ccp_alpha=hp.CLS.GB.CCP_ALPHA):
    clf = GradientBoostingClassifier(loss=loss, \
                                     learning_rate=learning_rate, \
                                     n_estimators=n_estimators, \
                                     subsample=subsample, \
                                     criterion=criterion, \
                                     min_samples_split=min_samples_split, \
                                     min_samples_leaf=min_samples_leaf, \
                                     min_weight_fraction_leaf=min_weight_fraction_leaf, \
                                     max_depth=max_depth, \
                                     min_impurity_decrease=min_impurity_decrease, \
                                     init=init, \
                                     random_state=random_state, \
                                     max_features=max_features, \
                                     verbose=verbose, \
                                     max_leaf_nodes=max_leaf_nodes, \
                                     warm_start=warm_start, \
                                     validation_fraction=validation_fraction, \
                                     n_iter_no_change=n_iter_no_change, \
                                     tol=tol, \
                                     ccp_alpha=ccp_alpha)
    return clf

def get_lda_classifier(n_components=hp.CLS.LDA.N_COMPONENTS, \
                       priors=hp.CLS.LDA.PRIORS, \
                       shrinkage=hp.CLS.LDA.SHRINKAGE, \
                       store_covariance=hp.CLS.LDA.STORE_COVARIANCE, \
                       tol=hp.CLS.TOL, \
                       solver=hp.CLS.LDA.SOLVER, \
                       covariance_estimator=hp.CLS.LDA.COVARIANCE_ESTIMATOR):
    clf = LinearDiscriminantAnalysis(n_components=n_components, \
                                      priors=priors, \
                                      shrinkage=shrinkage, \
                                      store_covariance=store_covariance, \
                                      tol=tol, \
                                      solver=solver, \
                                      covariance_estimator=covariance_estimator)    
    return clf

if __name__ == "__main__":
    print(get_lda_classifier(n_components=34, priors=[0.5, 0.5], shrinkage=0.5, store_covariance=True, tol=0.0001, solver='svd').get_params())