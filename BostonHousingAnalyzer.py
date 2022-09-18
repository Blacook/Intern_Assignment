################################################################################
# BostonHousingAnalyzer.py
#
# Shuhei Fukami
# Created 09/08/2022
# Copyright (c) 2022, WALC Inc.
################################################################################
from array import array
import os
import sys
import csv

import numpy as np
import scipy as sp

import pandas as pd
#import pandas_profiling as pdp

import sklearn
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate, cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import svm 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor


import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import HyperParamTuner as hpt


class Housings:
    """
    Class for analyzing the Bonston Housings data
    """
    def __init__(self, target:str="MEDV"):
        """
        To construct data required for predicting the price of Boston Housing
        Args:
            target(string) : a column of the data you wanna predict
        Returns:
            None
        """
        self.target = target
        boston = load_boston()
        X_all = pd.DataFrame(boston.data, columns=boston.feature_names)
        y_all = pd.DataFrame(boston.target, columns=[target])
        self.df = pd.concat([X_all, y_all], axis=1) 


    def get_data(self):
        """
        To download the Boston Housing data from the sklearn database
        Args:
            None
        Returns:
            self.df : a dataframe of the Boston Housing data
        """
        return self.df


    def __split_into_train_test(self):    
        """
        To split data into train(75%) and test(25%) of X, Y
        Args:
            None
        Returns:
            df_train(DataFrame):
            df_test (DataFrame): 
        """ 
        df_train, df_test = train_test_split(self.df, random_state=0)
        
        return  df_train, df_test
    
    
    def __split_data(self):    
        """
        To split the dataframe into features and target columns; Xtrain, Ytrain, Xtest, Ytest
        Args:
            None
        Returns:
            df_Xtrain(DataFrame):
            df_Ytrain(DataFrame):
            df_Xtest (DataFrame): 
            df_Ytest (DataFrame):
        """ 
        df_train, df_test = self.__split_into_train_test()
       
        df_Xtrain = df_train.iloc[:,:-1].copy()
        df_Ytrain = df_train.iloc[:,-1:].copy()
        df_Xtest  = df_test.iloc[:,:-1].copy()
        df_Ytest  = df_test.iloc[:,-1:].copy()
        
        return  df_Xtrain, df_Ytrain, df_Xtest, df_Ytest  
    
    
    def get_fundamental_statistics(self):
        """
        To download the Boston Housing data from the sklearn database
        Args:
            None
        Returns:
            self.df.describe()(DataFrame): fundamental statistics of the Boston Housing data
        """
        return self.df.describe() 
    
    
    def show_data_shape_summary(self):
        """
        To show the summary of the data in order to comfirm
        Args:
            None
        Returns:
            df_Xtrain(DataFrame):
            df_Ytrain(DataFrame): 
            df_Xtest (DataFrame):
            df_Ytest (DataFrame):
        """ 
        # check the number of rows and columns
        print(f"the number of rows and columns:{self.df.shape}")
        # check if any duplicate rows are present
        print(f"the number of any duplicate rows:{self.df.duplicated().sum()}")
        # check for NaN values
        print(f"the number of NaN values: \n {self.df.isna().sum()}")
        
        df_Xtrain, df_Ytrain, df_Xtest, df_Ytest = self.__split_data()
        print("X_train shape: {}".format(df_Xtrain.shape)) 
        print("y_train shape: {}".format(df_Ytrain.shape))
        print("X_test  shape: {}".format(df_Xtest.shape)) 
        print("Y_test  shape: {}".format(df_Ytest.shape))
        
        return df_Xtrain, df_Ytrain, df_Xtest, df_Ytest


    def show_data_profiling(self):
        """
        to show extraplatory data analysis profile and save the report as a html file
        Args:
            None
        Returns:
            df_train(DataFrame):
            df_test (DataFrame):
        """
        df_train, df_test = self.__split_into_train_test()
        # profiling the train dataframe (it takes too much time)
        #report = pdp.ProfileReport(df_train, title='Pandas Profiling Report', explorative=True, vars={"num": {"low_categorical_threshold": 0}} )
        #report.to_file('./info/profiling_report.html')
        
        return df_train, df_test
        
        
    #Show  
    def show_heatmaps_by_corr(self, df_train:pd.Series, fig_name:str='./info/corr_mat.png', fig_size:int=10):
        """
        to show heatmaps of correlation coeficients in each pair of two variables
         Args:
            df_train(DataFrame):
            fig_name:
            fig_size:
        Returns:
            None
        """
        plt.figure(figsize=(fig_size,fig_size))
        sns.heatmap(df_train.corr(), annot=True)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300)
        plt.show()         
        
            
    def scale_data(self, df_Xtrain:pd.Series, df_Xtest:pd.Series, scaler=Normalizer() ):
        """
        to scale the data  into an appropriate range
        Args:
            df_Xtrain(DataFrame):
            df_Xtest (DataFrame):
            scaler   (function) : You can choose a scaler from StandardScaler(), RobustScaler(), MinMaxScaler() and Normalizer(). Defaults to Normalizer().
        Returns:
            df_scaled_Xtrain(Dataframe):
            df_scaled_Xtest (Dataframe):
        """
        scaler = scaler.fit(df_Xtrain)
        
        # sacle each variable of the X data and fill the mean of the data if some defects exixst
        df = pd.DataFrame(scaler.transform(df_Xtrain), columns=df_Xtrain.columns, dtype='float' )
        df_scaled_Xtrain = df.fillna(df.mean())
        df = pd.DataFrame(scaler.transform(df_Xtest), columns=df_Xtest.columns, dtype='float')
        df_scaled_Xtest = df.fillna(df.mean())

        return df_scaled_Xtrain, df_scaled_Xtest
        

    def select_feature_data(self, df_train:pd.Series, df_test:pd.Series, corr_threshold:float=0.5):
        """
        = Comparison with the correlation coefficients ==
        Args:
            df_train   (DataFrame):
            df_test    (DataFrame):
            corr_threshold (float): _description_. Defaults to 0.5.
        Returns:
            selected_Xtrain(DataFrame):
            selected_Xtest (DataFrame):
        """
        # the absolute threshold of coorelation coefficients with the target value
        # compare absolute correlation coefficients with y(MEDV) to the threshhold 
        corr_matrix = df_train.corr()
        # display(corr_matrix)
        corr_y = pd.DataFrame({"features":df_train.columns,"corr_y":corr_matrix[self.target]},index=None)
        corr_y = corr_y.reset_index(drop=True)
        corr_y.style.background_gradient()
        select_cols = corr_y[corr_y["corr_y"].abs()>corr_threshold]
        
        #display("correlation coefficients of feature variables with the target:", select_cols)

        select_cols = list(select_cols["features"])

        #data after choosing feature variables
        df_train_new = df_train.loc[:,select_cols]
        df_test_new = df_test.loc[:,select_cols]
        selected_Xtrain = df_train_new.iloc[:,:-1].copy()
        selected_Xtest = df_test_new.iloc[:,:-1].copy()
        
        return selected_Xtrain, selected_Xtest
    
        #display(df_train_new.head())
        #display(self.selectedXtrain.head())
        #show heatmap
        #fig_name = './info/features_corr_mat_' + str(corr_threshold) + '.png'
        #self.__show_heatmaps_by_corr(fig_name=fig_name)
        
        #show scatter plots with feature variables
        #sns.set(style='whitegrid', context='notebook')
        #sns.pairplot(df_train_new, size=2.5)
        #plt.tight_layout()
        #plt.savefig('./info/features_scatter_corr_' + str(corr_threshold) + '.png', dpi=300)
        #plt.show()
        

    def princpal_components_analysis(self, df_Xtrain:pd.Series, df_Xtest:pd.Series, n_components=2, scaler=Normalizer() ):

        df_scaled_Xtrain, df_scaled_Xtest = self.scale_data(df_Xtrain, df_Xtest, scaler=scaler)

        print("========= Princpal Components Analysis =========")
        pca_list = ["First component", "Second component"]#, "Third component", "Forth component", "Fifth component"]
        #maintain the primary &secondary componets of the data
        pca = PCA(n_components=n_components).fit(df_scaled_Xtrain)

        #print("PCA components:\n{}".format(pca.components_))
        #plot the heatmap of the coefficients
        plt.matshow(pca.components_ , cmap='viridis')
        plt.yticks([0, 1], pca_list)
        plt.colorbar()
        plt.xticks(range(len(df_scaled_Xtrain.columns)), df_scaled_Xtrain.columns, rotation=60, ha='left')
        plt.xlabel("Feature") 
        plt.ylabel("Principal components")
        plt.savefig('./info/features_PCA.png', dpi=300)

        # output the contribution ratio of principal components
        print('contribution ratio  of each conponent: {0}'.format(pca.explained_variance_ratio_))
        print('accumulated contribution ratio: {0}'.format(sum(pca.explained_variance_ratio_)))
        
        # transform the dataset to principal components
        df_pca_Xtrain = pd.DataFrame(pca.transform(df_scaled_Xtrain), columns=pca_list, dtype='float' )
        df_pca_Xtest = pd.DataFrame(pca.transform(df_scaled_Xtest), columns=pca_list, dtype='float' )

        return df_pca_Xtrain, df_pca_Xtest 


    def plot_pca(self, n_components=2):
        scaler=StandardScaler()
        X, y = self.GetXy()
        X_scaled = scaler.fit_transform(X)
        pca=PCA(n_components=n_components) 
        X_pca = pca.fit_transform(X_scaled)
        
        # plot 
        plt.figure(figsize=(8, 8))
        mglearn.discrete_scatter(X_pca[:,0],X_pca[:, 1], self.df[self.Names_y])
        plt.legend(self.Names_Target, loc="best")
        plt.gca().set_aspect("equal") 
        plt.xlabel("First principal compornent")
        plt.ylabel("Second principal compornent")
        
        plt.matshow(pca.components_, cmap="viridis")
        plt.yticks([0,1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.Names_X)), self.Names_X, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("Principal components")
        
        return pd.DataFrame(X_scaled, columns=self.Names_X), pd.DataFrame(X_pca),pca        
        
       
    def __preprocess_train_test(self, scaled:bool=True, PCA:bool=True):
        df_train, df_test = self.__split_into_train_test() 
        df_Xtrain, df_Ytrain, df_Xtest, df_Ytest = self.__split_data()
        
        if scaled==False and PCA==False:
            df_Xtrain = df_Xtrain
            df_Xtest = df_Xtest
        elif scaled==True and PCA==False:
            df_Xtrain, df_Xtest = self.scale_data(df_Xtrain, df_Xtest, scaler)
        elif scaled==True and PCA==True:
            df_Xtrain, df_Xtest = self.princpal_components_analysis(df_Xtrain, df_Xtest, n_components, scaler)
        else:
            df_Xtrain, df_Xtest = self.select_feature_data(df_train, df_test, corr_threshold) 
        
        return df_Xtrain, df_Ytrain, df_Xtest, df_Ytest
     
     
    def check_valid_score(model:dict, X_train, Y_train, X_valid, Y_valid): 
            # build models
            model.fit(X_train, Y_train)
            # evaluate models
            Y_pred = model.predict(X_valid) 
            #print("Test set predictions:In{}".format(Y_pred))
            #print("Test set truths:In{}".format(Y_valid))
            valid_score = mean_squared_error(Y_valid, Y_pred, squared=False, multioutput='uniform_average' )
            #print("Valid set score: {:.4f}".format(valid_score))
            return valid_score
         
                  
    def cross_validation(model, df_Xtrain:pd.Series, df_Xtest:pd.Series, n_valid:int, scaled:bool, PCA:bool):# the number of iteration for validation
            test_scores = []
            #Cross Validation
            CV = KFold(n_splits=n_valid, shuffle=True, random_state=0)
            
            df_Ytrain = self.df_Ytrain
            df_Ytest = self.df_Ytest
                
            num_CV = 0
            for fold_idx, (train_idx, valid_idx) in enumerate(CV.split(df_Xtrain, df_Ytrain)):
                #print(num_CV)
                X_train = df_Xtrain.iloc[train_idx, :]
                X_valid = df_Xtrain.iloc[valid_idx, :]
                Y_train = df_Ytrain.iloc[train_idx, :]
                Y_valid = df_Ytrain.iloc[valid_idx, :]
                
                valid_score = check_valid_score(model, X_train, Y_train, X_valid, Y_valid)
                #print(valid_score)
                Y_pred = model.predict(df_Xtest)
                Y_true = df_Ytest
                # cakcurate RSME
                test_score = mean_squared_error(Y_true, Y_pred, squared=False, multioutput='uniform_average' )
                
                test_scores.append(test_score)
                num_CV += 1
            
            
            test_score_mean = np.mean(test_scores)
            test_score_std = np.std(test_scores)
            print("test_scores(RMSE) for \n" + str(model))# + ": \n", test_scores)
            print("mean of test_scores:" + str(test_score_mean) , "std of test_scores:" + str(test_score_std))
            return test_scores
         
         
    def AllSupervised(self, n_valid:int=5, scaled:bool=True, PCA:bool=True):   
        dict_scores = {}
        print("============= Linear Regression =============")
        params = {#'fit_intercept':True, 
                  #'normalize':False, 
                  #'copy_X':True,
                  #'n_jobs':None
                 }
        model = sklearn.linear_model.LinearRegression(**params)
        dict_scores['Linear']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        
        print("======== Ridge(L2) Linear Regression ========")
        params = {'alpha': 0.3,
                  'max_iter': 20,
                  'tol': 1e4, 
                  'random_state': None
                  #'fit_intercept':True, 
                  #'normalize':False, 
                  #'copy_X':True,
                 }
        model = sklearn.linear_model.Ridge(**params)
        dict_scores['Ridge']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("======== Lasso(L1) Linear Regression ========")
        params = {'alpha': 0.3, 
                  'max_iter': 10000, 
                  'tol': 1e-4, 
                  'warm_start': False, 
                  'positive': False, 
                  'random_state': None, 
                  'selection': "cyclic"
                  #'fit_intercept':True, 
                  #'normalize':False, 
                  #'copy_X':True, 
                  }
        model = sklearn.linear_model.Lasso(**params)
        dict_scores['Lasso']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("==== Elastic Net(L1&L2) Linear Regression ====")
        params = {'alpha': 0.1,
                  'l1_ratio': 0.5,
                  'max_iter': 10000,
                  'tol': 0.001,
                  'warm_start': False,
                  'positive': False,
                  'random_state': None,
                  'selection': "cyclic"
                  #'fit_intercept':True, 
                  #'normalize':False, 
                  #'copy_X':True, 
                  }
        model = sklearn.linear_model.ElasticNet(**params)
        dict_scores['Elastic Net']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("========== k-neighbors Regression  ==========")
        params = {'n_neighbors': 10
                  }
        model = KNeighborsRegressor(**params)
        dict_scores['KNN']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("========== Supprot Vector Regression  ==========")
        params = {'C': 0.005, 
                  'kernel': 'sigmoid',
                  'epsilon': 1e-05, 
                  'gamma': 'auto'
                 }
        model = svm.SVR(**params)
        dict_scores['SVR']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("========== Decision Tree Regression ==========")
        params = {'criterion': 'squared_error', 
                  'splitter': 'best', 
                  'max_depth': 20, 
                  'min_samples_split': 2, 
                  'min_samples_leaf': 1, 
                  'min_weight_fraction_leaf': 0.0,
                  'max_features': None, 
                  'random_state': None, 
                  'max_leaf_nodes': None, 
                  'min_impurity_decrease': 0.0, 
                  'ccp_alpha':0.0
                  }
        model = DecisionTreeRegressor(**params)
        dict_scores['Regression Tree']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        print("========== Random Forest Regression ==========")
        params = {'bootstrap': True, 
                  'criterion': 'squared_error',
                  'max_depth': None,
                  'max_features': 'auto', 
                  'max_leaf_nodes': None,
                  'min_samples_leaf': 1,
                  'min_samples_split': 2, 
                  'min_weight_fraction_leaf': 0.0,
                  'n_estimators': 10, 
                  'n_jobs': 1, 
                  'oob_score': False, 
                  'random_state': None,
                  'verbose': 0, 
                  'warm_start': False
                  }
        model = RandomForestRegressor(**params)
        dict_scores['Random Forest']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        
        print("============== LGBM Regression ==============")
        params = {'learning_rate': 0.2,
                  'objective':'regression', 
                  'metric':'mse',
                  'num_leaves': 50,
                  'verbose': -1,
                  #'bagging_fraction': 0.7,
                  #'feature_fraction': 0.7,
                  #'force_col_wise': True,
                  #'colsample_bytree' :1.0,
                  'subsample': 1.0,
                  'n_estimators': 150,
                  }
        model = LGBMRegressor(**params)
        dict_scores['LGBM']  = np.mean(cross_validation(model, n_valid, scaled, PCA))
        
        
        print("=========== the best algorithms ===========")
        #display(sorted(dict_scores.items(), key=lambda x:x[1]))

         
    def HyperPrameterTuning(self, n_trials:int=100, n_valid:int=5, scaled:bool=True, PCA:bool=True): 
        df_Xtrain, df_Ytrain, df_Xtest, df_Ytest = self.__preprocess_train_test(scaled, PCA)
        X = pd.concat([df_Xtrain, df_Xtest])   
        y = pd.concat([df_Ytrain, df_Ytest])
        y = sklearn.utils.validation.column_or_1d(y, warn=True) 
        
        sampler = optuna.samplers.RandomSampler(seed=10)
        study = optuna.create_study(direction="maximize", sampler=sampler)# maximize negative RSME
        study.optimize(objective_auguments(X, y, n_valid), n_trials=n_trials)
        # output the optimal solution
        print(f"The best value is : \n {study.best_value}")
        print(f"The best parameters are : \n {study.best_params}")
        optuna.visualization.matplotlib.plot_optimization_history(study)
        optuna.visualization.matplotlib.plot_param_importances(study)