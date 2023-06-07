import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,mean_squared_error

def is_numpy_array_in_list_of_numpy_arrays(numpy_array_to_check, list_of_numpy_arrays):
  '''
  A function to check if a numpy array is in a list of numpy arrays
  '''
  for numpy_array in list_of_numpy_arrays:
    if np.array_equal(numpy_array_to_check, numpy_array):
      return True
  return False

class Feature_Selection_PSO:
    '''
    A class to perform feature selection using Particle Swarm Optimization
    '''
    def __init__(self,X,Y,label_type,iterations=50,swarm_size=30,verbose=False):
        '''
        Constructor for Feature_Selection_PSO class

        Arguments:
        X -- A numpy array or pandas dataframe of features
        Y -- A numpy array or pandas dataframe of labels
        label_type -- A string to specify the type of labels. Either 'Classification' or 'Regression'
        iterations -- An integer specifying the number of iterations to run the algorithm for
        swarm_size -- An integer specifying the number of particles in the swarm. If None, the swarm size is set to half the number of possible particles
        verbose -- A boolean to specify whether to print the progress of the algorithm
        '''

        self.__c1, self.__c2 = 2,2
        self.__R1, self.__R2 = round(np.random.uniform(0,1),2), round(np.random.uniform(0,1),2)
        self.__X, self.__Y = X,Y
        self.__label_type = label_type
        self.__iterations = iterations
        self.__verbose = verbose
        self.__swarm_size = swarm_size if swarm_size else int(0.5 * (2 ** self.__X.shape[1]))
        self.__velocity = np.zeros((self.__swarm_size,self.__X.shape[1]))
        self.__pbest = np.zeros((self.__swarm_size,self.__X.shape[1]))
        self.__gbest = np.zeros((1,self.__X.shape[1]))
        self.__pbest_fitness = np.zeros((self.__swarm_size,1))
        self.__gbest_fitness = 0
        self.__fitness = np.zeros((self.__swarm_size,1))

    def __Make_Particle(self):
        '''
        A function to make a particle i.e. a random binary array of size equal to the number of features
        '''
        return np.random.randint(0,2,size=(self.__X.shape[1])).astype('float32')
    
    def __Make_Swarm(self):
        '''
        A function to make a swarm i.e. a list of particles
        '''
        swarm = []
        for i in range(self.__swarm_size):
            particle = self.__Make_Particle()

            #check if particle is already in swarm
            while is_numpy_array_in_list_of_numpy_arrays(particle,swarm):
                particle = self.__Make_Particle()

            #check if particle is all 0
            while np.sum(particle) == 0:
                particle = self.__Make_Particle()

            swarm.append(particle)
        self.__swarm = np.array(swarm)
        self.__pbest = deepcopy(self.__swarm)

    def __Fitness_Classification(self,actual,predicted):
        '''
        A function to calculate the fitness of a particle for a classification problem
        '''
        F1 = f1_score(actual,predicted,average='macro') * 100
        return F1
    
    def __Fitness_Regression(self,actual,predicted):
        '''
        A function to calculate the fitness of a particle for a regression problem
        '''
        RMSE = mean_squared_error(actual,predicted) ** 0.5
        return RMSE

    def __Evaluate_Classification(self,particle):
        '''
        A function to evaluate a particle for a classification problem
        '''
        particle = np.array(particle)
        X = self.__X if isinstance(self.__X,pd.DataFrame) else pd.DataFrame(self.__X)
        Y = self.__Y

        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        model = RandomForestClassifier()
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        return self.__Fitness_Classification(y_test,y_pred)
    
    def __Evaluate_Regression(self,particle):
        '''
        A function to evaluate a particle for a regression problem
        '''
        particle = np.array(particle)
        X = self.__X if isinstance(self.__X,pd.DataFrame) else pd.DataFrame(self.__X)
        Y = self.__Y
        
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        model = RandomForestRegressor()
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        return self.__Fitness_Regression(y_test,y_pred)
    
    def __Evaluate(self,particle):
        if self.__label_type == 'Classification':
            return self.__Evaluate_Classification(particle)
        elif self.__label_type == 'Regression':
            return self.__Evaluate_Regression(particle)
        else:
            raise Exception('Invalid Label Type. Please choose either Classification or Regression')
        
    def __Update_Pbest(self):
        '''
        A function to update the personal best of each particle
        '''
        for i in range(self.__swarm_size):
            if (self.__label_type == 'Classification') and (self.__pbest_fitness[i] < self.__fitness[i]):
                self.__pbest[i] = deepcopy(self.__swarm[i])
                self.__pbest_fitness[i] = deepcopy(self.__fitness[i])
            elif (self.__label_type == 'Regression') and (self.__pbest_fitness[i] > self.__fitness[i]):
                self.__pbest[i] = deepcopy(self.__swarm[i])
                self.__pbest_fitness[i] = deepcopy(self.__fitness[i])
    
    def __Update_Gbest(self):
        '''
        A function to update the global best particle
        '''
        if self.__label_type == 'Classification':
                self.__gbest = deepcopy(self.__pbest[np.argmax(self.__pbest_fitness)])
                self.__gbest_fitness = deepcopy(np.max(self.__pbest_fitness))
        elif self.__label_type == 'Regression':
                self.__gbest = deepcopy(self.__pbest[np.argmin(self.__pbest_fitness)])
                self.__gbest_fitness = deepcopy(np.min(self.__pbest_fitness))
        
    def __Update_Swarm(self):
        '''
        A function to update the swarm's position
        '''
        for i in range(self.__swarm_size):
            self.__swarm[i] += self.__velocity[i]
    
    def __Calculate_New_Velocity(self):
        '''
        A function to calculate the new velocity of each particle

        Velocity = (c1 * R1 * (pbest - current_position)) + (c2 * R2 * (gbest - current_position))
        '''
        for i in range(self.__swarm_size):
            self.__velocity[i] += ((self.__c1 * self.__R1 * (self.__pbest[i] - self.__swarm[i])) + (self.__c2 * self.__R2 * (self.__gbest - self.__swarm[i])))

    def __Update(self,):
        '''
        A function to update the swarm for each iteration
        '''
        self.__Calculate_New_Velocity()
        self.__Update_Swarm()
        self.__Update_Pbest()
        self.__Update_Gbest()

    def __Print(self,iteration):
        '''
        A function to print the iteration and the global best fitness
        '''
        if self.__verbose:
            print('Iteration: ',iteration,' Gbest Fitness: ',self.__gbest_fitness)
    
    def __Run(self):
        '''
        A function to run the algorithm
        '''
        self.__Make_Swarm()
        Track_Record_Of_Gbest_Fitness = []
        for i in range(self.__iterations):
            for j in range(self.__swarm_size):
                Data = self.__swarm[j]
                Data = np.where(Data > 1,1,Data)
                Data = np.where(Data < 0,0,Data)
                Data = np.where(Data > 0.5,1,0)

                self.__fitness[j] = self.__Evaluate(Data)
             
            if i == 0:
                self.__gbest = deepcopy(self.__swarm[np.argmax(self.__fitness)])
                self.__gbest_fitness = deepcopy(np.max(self.__fitness))

            self.__Update()
            self.__Print(i)

            #track record of gbest fitness
            Track_Record_Of_Gbest_Fitness.append(self.__gbest_fitness)

            #if change in previous 5 gbest fitness is less than 0.1% then stop
            if (i > self.__iterations//3) and (np.std(Track_Record_Of_Gbest_Fitness[-5:]) < 0.1):
                return

    def __Get_Features(self):
        '''
        A function to get the selected features
        '''
        if type(self.__X) == pd.core.frame.DataFrame:
            gbest = np.array(self.__gbest)
            gbest[gbest > 1] = 1
            gbest[gbest < 0] = 0
            gbest = np.round(gbest)
            indices = np.where(gbest==1)
            print('Following are the selected features: ',self.__X.columns[indices].tolist())
            return self.__X.columns[indices].tolist()
        
        elif type(self.__X) == np.ndarray:
            gbest = np.array(self.__gbest)
            gbest[gbest > 1] = 1
            gbest[gbest < 0] = 0
            gbest = np.round(gbest)
            indices = np.where(gbest==1)
            print('Following are the indices of selected features: ',indices[0])
            return indices[0]

    def __Get_Fitness(self):
        '''
        A function to get the fitness by the selected features
        '''
        return self.__gbest_fitness
    
    def Get_All_Data(self):
        '''
        A function to get all the data
        '''
        print('Swarm: ',self.__swarm)
        print('Pbest: ',self.__pbest)
        print('Gbest: ',self.__gbest)
        print('Pbest Fitness: ',self.__pbest_fitness)
        print('Gbest Fitness: ',self.__gbest_fitness)
        print('Fitness: ',self.__fitness)
        print('Velocity: ',self.__velocity)
    
    def Main(self):
        '''
        Main function to run the algorithm
        '''
        self.__Run()
        return self.__Get_Features(),self.__Get_Fitness()