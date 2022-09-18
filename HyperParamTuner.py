################################################################################
# BostonHousingAnalyzer.py
#
# Shuhei Fukami
# Created 09/08/2022
# Copyright (c) 2022, WALC Inc.
################################################################################

import numpy as np
import optuna

import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate, cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

import BostonHousingAnalyzer 


def objective_auguments(X, y, n_valid):
    def objective(trial): 
            regressor_name = trial.suggest_categorical("regressor", 
                                                        [#"Linear",
                                                         "Ridge",
                                                         "Lasso",
                                                         #"Elastic Net",
                                                         "kNN",
                                                         #"SVR",
                                                         "Regression Tree",
                                                         #"Random Forest",
                                                         "LGBM"
                                                        ]
                                                       )


            if regressor_name == 'Linear':
                print("============= Linear Regression =============")
                linear_n_job = trial.suggest_int("linear_n_job",
                                            1, 100,
                                            log=True
                                           )
                params = {#'fit_intercept':True, 
                      #'normalize':False, 
                      #'copy_X':True,
                      'n_jobs': linear_n_job
                     }
                regressor_obj = sklearn.linear_model.LogisticRegression(**params)


            elif regressor_name == "Ridge":
                print("======== Ridge(L2) Linear Regression ========")
                L2_alpha = trial.suggest_float("L2_alpha",
                                            1e-5, 1e5,
                                            log=True
                                           )
                L2_max_iter = trial.suggest_int("L2_max_iter",
                                            10, 1e5,
                                            log=True
                                           )
                L2_tol = trial.suggest_float("L2_tol",
                                            1e-10, 1e5,
                                            log=True
                                           )
                params = {'alpha': L2_alpha,
                      'max_iter': L2_max_iter,
                      'tol': L2_tol, 
                      'random_state': None
                      #'fit_intercept':True, 
                      #'normalize':False, 
                      #'copy_X':True,
                     }

                regressor_obj = sklearn.linear_model.Ridge(**params)


            elif regressor_name == "Lasso":
                print("======== Lasso(L1) Linear Regression ========")
                L1_alpha = trial.suggest_float("L1_alpha",
                                            1e-5, 1e5,
                                            log=True
                                           )
                L1_max_iter = trial.suggest_int("L1_max_iter",
                                            10, 1e5,
                                            log=True
                                           )
                L1_tol = trial.suggest_float("L1_tol",
                                            1e-10, 1e5,
                                            log=True
                                           )
                params = {'alpha': L1_alpha, 
                      'max_iter': L1_max_iter, 
                      'tol': L1_tol, 
                      'warm_start': False, 
                      'positive': False, 
                      'random_state': None, 
                      'selection': "cyclic"
                      #'fit_intercept':True, 
                      #'normalize':False, 
                      #'copy_X':True, 
                      }

                regressor_obj = sklearn.linear_model.Lasso(**params)


            elif regressor_name == "Elastic Net":
                print("==== Elastic Net(L1&L2) Linear Regression ====")
                L1L2_alpha = trial.suggest_float("L1L2_alpha",
                                            1e-5, 1e5,
                                            log=True
                                           )
                L1L2_ratio = trial.suggest_float("L1L2_ratio",
                                            1e-10, 1,
                                            log=True
                                           )
                L1L2_max_iter = trial.suggest_int("L1L2_max_iter",
                                            10, 1e5,
                                            log=True
                                           )
                L1L2_tol = trial.suggest_float("L1L2_tol",
                                            1e-10, 1e5,
                                            log=True
                                           )
                params = {'alpha': L1L2_alpha,
                      'l1_ratio': L1L2_ratio,
                      'max_iter': L1L2_max_iter,
                      'tol': L1L2_tol,
                      'warm_start': False,
                      'positive': False,
                      'random_state': None,
                      'selection': "cyclic"
                      #'fit_intercept':True, 
                      #'normalize':False, 
                      #'copy_X':True, 
                      }

                regressor_obj = sklearn.linear_model.ElasticNet(**params)


            elif regressor_name == "kNN":
                print("========== k-neighbors Regression  ==========")
                kNN_n_neighbors = trial.suggest_int("kNN_n_neighbors",
                                            1, 10,
                                           )
                params = {'n_neighbors': kNN_n_neighbors
                      }

                regressor_obj = KNeighborsRegressor(**params)


            elif regressor_name == "SVR":
                print("========= Support Vector Regressor =========")
                SVR_c = trial.suggest_float("SVR_c",
                                            1e-5, 1e5,
                                            log=True
                                           )
                SVR_kernel = trial.suggest_categorical("SVR_kernel", 
                                                        ["linear", 
                                                         "poly", 
                                                         "rbf", 
                                                         "sigmoid"
                                                        ]
                                                       )
                SVR_epsilon = trial.suggest_float("SVR_epsilon",
                                            1e-10, 10,
                                            log=True
                                           )
                params = {'C': SVR_c, 
                      'kernel': SVR_kernel,
                      'epsilon': SVR_epsilon, 
                      'gamma': 'auto'
                     }
                regressor_obj = svm.SVR(**params)

            elif regressor_name == "Regression Tree":
                print("========= Decision Tree Regressor =========")
                rt_max_depth = trial.suggest_int("rt_max_depth",
                                            1, 20,
                                           )
                params = {'criterion': 'squared_error', 
                      'splitter': 'best', 
                      'max_depth': rt_max_depth, 
                      'max_features': 1.0,
                      'max_leaf_nodes': None,
                      'min_samples_leaf': 2, 
                      'min_samples_split': 2, 
                      'min_weight_fraction_leaf': 0.1, 
                      'min_impurity_decrease': 0.0, 
                      'random_state': None, 
                      'ccp_alpha':0.0
                      }

                regressor_obj = DecisionTreeRegressor(**params)


            elif regressor_name == "Random Forest":
                print("========== Random Forest Regression ==========")
                rf_n_estimators = trial.suggest_int("rf_n_estimators",
                                                    10, 100,
                                                    log=True
                                                   )
                rf_max_depth = trial.suggest_int("rf_max_depth", 
                                                 2, 30,
                                                 log=True
                                                )
                params = {'bootstrap': True, 
                      'criterion': 'squared_error',
                      'max_depth': rf_max_depth,
                      'max_features': 1.0, 
                      'max_leaf_nodes': None,
                      'min_samples_leaf': 2,
                      'min_samples_split': 2, 
                      'min_weight_fraction_leaf': 0.1,
                      'n_estimators': rf_n_estimators, 
                      'n_jobs': 1, 
                      'oob_score': False, 
                      'random_state': None,
                      'verbose': -1, 
                      'warm_start': False
                      }

                regressor_obj = RandomForestRegressor(**params)

            else: 
                print("============== LGBM Regression ==============")
                lgbm_learning_rate = trial.suggest_float("lgbm_learning_rate",
                                                    1e-6, 1e2,
                                                    log=True
                                                   )
                lgbm_n_estimators = trial.suggest_int("lgbm_n_estimators",
                                                    10, 1000,
                                                    log=True 
                                                   )
                lgbm_num_leaves = trial.suggest_int("lgbm_num_leaves", 
                                                 2, 100,
                                                 log=True
                                                )
                params = {'learning_rate': lgbm_learning_rate,
                      'objective':'regression', 
                      'metric':'mse',
                      'num_leaves': lgbm_num_leaves,
                      #'bagging_fraction': 0.7,
                      #'feature_fraction': 0.7,
                      #'force_col_wise': True,
                      #'colsample_bytree' :1.0,
                      'subsample': 1.0,
                      'n_estimators': lgbm_n_estimators,
                      'verbose': -1,
                      }
                regressor_obj = LGBMRegressor(**params)


            scores = cross_validate(regressor_obj,
                                    X, 
                                    y,
                                    scoring='neg_root_mean_squared_error', 
                                    error_score='raise',
                                    n_jobs=-1, 
                                    cv=n_valid)
            # the number of pra-processing jobs（-1 means to use the all processors）
            RSME = scores['test_score'].mean()
            
            return RSME
        
    return objective
