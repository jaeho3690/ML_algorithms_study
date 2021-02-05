import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

class PlotArtificialData:
    def __init__(self,constraint_type=1,seed=2,data_size=1000,plot=True):
        self.data_size= data_size
        self._plot = plot
        self.X_grid = np.linspace(-7,7,500)
        self.y_grid = np.linspace(-7,7,500)
        self._cluster_num =2
        self._seed = seed
        # This epsilon is used so that the covariance matrix is non-singular
        self._epsilon = 1e-7 * np.eye(2)
        # Set seed
        np.random.seed(self._seed)
        if constraint_type==1:
            self._df = self.balanced_spherical()
        elif constraint_type==2:
            self._df =  self.imbalanced_spehrical()
        elif constraint_type==3:
            self._df =  self.balanced_nonspherical()
        else:
            self._df =  self.imbalanced_nonspheric_variance()
    def balanced_spherical(self):
        balanced_data_size = int(self.data_size/2)
        mean1= np.random.choice(self.X_grid,2)
        mean2= np.random.choice(self.X_grid,2)
        variance1 = np.random.choice(np.arange(1,5),1)
        variance2 = np.random.choice(np.arange(1,5),1)
        variance3 = np.random.choice(np.arange(1,5),1)
        variance4 = np.random.choice(np.arange(1,5),1)
        print(f'Data size: {balanced_data_size}')
        x1, y1 = np.random.multivariate_normal(mean1,np.array([[variance1,0],[0,variance2]]), balanced_data_size).T
        x2, y2 = np.random.multivariate_normal(mean2,np.array([[variance3,0],[0,variance4]]), balanced_data_size).T
        
        if self._plot:
            self.plot(x1,y1,x2,y2)
        return self.to_dataframe(x1,y1,x2,y2)
    
    def imbalanced_spehrical(self):
        imbalance_ratio=0.9
        data_size1 = int(self.data_size*imbalance_ratio)
        data_size2 = int(self.data_size*(1-imbalance_ratio))
        mean1= np.random.choice(self.X_grid,2)
        mean2= np.random.choice(self.X_grid,2)
        variance1 = np.random.choice(np.arange(1,5),1)
        variance2 = np.random.choice(np.arange(1,5),1)
        variance3 = np.random.choice(np.arange(1,5),1)
        variance4 = np.random.choice(np.arange(1,5),1)
        print(f'Data 1: {data_size1}, Data 2: {data_size2}')
        x1, y1 = np.random.multivariate_normal(mean1,np.array([[variance1,0],[0,variance2]]), data_size1).T
        x2, y2 = np.random.multivariate_normal(mean2,np.array([[variance3,0],[0,variance4]]), data_size2).T
        
        if self._plot:
            self.plot(x1,y1,x2,y2)
        return self.to_dataframe(x1,y1,x2,y2)
    def balanced_nonspherical(self):
        balanced_data_size = int(self.data_size/2)
        mean1= np.random.choice(self.X_grid,2)
        mean2= np.random.choice(self.X_grid,2)
        covariance1 = np.random.choice(np.arange(3,7),1)[0]
        covariance2 = np.random.choice(np.arange(3,7),1)[0]
        variance =  np.random.choice(np.arange(5,10),1)
        variance2 = np.random.choice(np.arange(5,10),1)
        print(f'Data size: {balanced_data_size} Each, covariance1 :{covariance1}, covariance2 :{covariance2}, variance : {variance[0]}')
        x1, y1 = np.random.multivariate_normal(mean1,np.array([[variance,covariance1],[covariance1,variance]])+self._epsilon, balanced_data_size).T
        x2, y2 = np.random.multivariate_normal(mean2,np.array([[variance2,covariance2],[covariance2,variance2]])+self._epsilon, balanced_data_size).T
        
        if self._plot:
            self.plot(x1,y1,x2,y2)  
        return self.to_dataframe(x1,y1,x2,y2)
    def imbalanced_nonspheric_variance(self):
        imbalance_ratio=0.9
        data_size1 = int(self.data_size*imbalance_ratio)
        data_size2 = int(self.data_size*(1-imbalance_ratio))
        mean1= np.random.choice(self.X_grid,2)
        mean2= np.random.choice(self.X_grid,2)
        covariance1 = np.random.choice(np.arange(1,5),1)[0]
        covariance2 = np.random.choice(np.arange(1,5),1)[0]
        variance =  np.random.choice(np.arange(5,10),1)
        variance2 = np.random.choice(np.arange(5,10),1)
        print(f'Data 1: {data_size1}, Data 2: {data_size2}, covariance:{covariance1}')
        x1, y1 = np.random.multivariate_normal(mean1,np.array([[variance,covariance1],[covariance1,variance]])+self._epsilon, data_size1).T
        x2, y2 = np.random.multivariate_normal(mean2,np.array([[variance2,covariance2],[covariance2,variance2]])+self._epsilon, data_size2).T
        
        if self._plot:
            self.plot(x1,y1,x2,y2)
        return self.to_dataframe(x1,y1,x2,y2)
        
    def to_dataframe(self,x1,y1,x2,y2):
        data1 = pd.DataFrame({'col1':x1,'col2':y1,'label':1})
        data2 = pd.DataFrame({'col1':x2,'col2':y2,'label':2})
        return pd.concat([data1,data2],ignore_index=True)
    
    def plot(self,x1,y1,x2,y2):
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.plot(x1, y1, 'o',color='r')
        plt.plot(x2, y2, 'o',color='c')
        plt.axis('equal')