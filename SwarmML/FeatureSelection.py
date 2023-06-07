import numpy as np
import pandas as pd
from copy import deepcopy
from builtins import Exception

from .AlgorithmPSO import *

def Drop_Nulls(X,Y):
    '''
    A function to drop all rows with null values in X and Y
    '''
    null_indices = np.where(pd.isnull(X))
    X = np.delete(X,null_indices[0],axis=0)
    Y = np.delete(Y,null_indices[0],axis=0)
    return (X,Y)

class Particle_Swarm_Optimization:
    '''
    A class to implement Particle Swarm Optimization Algorithm
    '''
    def __init__(self,label_type,verbose=False):
        '''
        Constructor to initialize the parameters
        '''
        self.__Validate_Label_Type(label_type)
        self.label_type = label_type
        self.__Validate_Verbose(verbose)
        self.__verbose = verbose

    def __Validate_Verbose(self,verbose):
        '''
        A function to validate the verbose parameter
        '''
        if type(verbose) != bool or verbose not in [True,False]:
            raise Exception('Verbose should be boolean. Only True or False are valid.')

    def __Validate_Label_Type(self,label_type):
        '''
        A function to validate the label type
        '''
        if label_type not in ['Classification','Regression']:
            raise Exception('Invalid Label Type. Only Classification or Regression are valid.')
        
    def __Validate_Data(self,X,Y):
        '''
        A function to validate the data
        '''
        Validity_Check_DataSamples = self.__Num_DataPoints_in_X(X)
        Validity_Check_DataSamples = self.__Num_DataPoints_Equal_in_X_and_Y(X,Y)
        if Validity_Check_DataSamples:
            X,Y = self.__Nulls_in_X(X,Y)
        return X,Y
    
    def __Num_DataPoints_Equal_in_X_and_Y(self,X,Y):
        '''
        A function to check if the number of data points in X and Y are equal
        '''
        if np.shape(X)[0] != np.shape(Y)[0]:
            raise Exception('Number of Data Points in X and Y are not equal. Cannot proceed with Feature Selection')
        else:
            return 1

    def __Num_DataPoints_in_X(self,X):
        '''
        A function to check if the number of data points in X are valid
        '''
        Num_DataPoints = np.shape(X)[0]
        if Num_DataPoints < 2:
            raise Exception('Training data should have minimum two data samples / rows . Cannot proceed with Feature Selection')
        else:
            return 1

    def __Nulls_in_X(self,X,Y):
        '''
        A function to check if there are any nulls in X
        '''
        if type(X) == None:
            raise Exception('Termination Called')
        else:
            if (type(X) == pd.DataFrame and X.isnull().sum().sum() > 0) or (type(X) == np.ndarray and np.isnan(X).sum().sum() > 0):
                print('Nulls in X. Dropping rows with Nulls')
                Updated_X,Updated_Y = Drop_Nulls(X,Y)
                return Updated_X,Updated_Y
            else:
                return X,Y
    
    def run(self,X,Y):
        '''
        A function to run the algorithm
        '''
        X,Y = self.__Validate_Data(X,Y)
        Best_Features,Best_Fitness = Feature_Selection_PSO(X,Y,self.label_type,iterations=50,swarm_size=30,verbose=self.__verbose).Main()
        return Best_Features,Best_Fitness