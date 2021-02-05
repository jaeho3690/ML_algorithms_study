import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap

class KMeans:
    """Implementation of Hard KMeans"""
    def __init__(self,K,plot=True,is_print=True):
        assert type(K) is int,"K should be Integer"
        self._K= K
        self.plot= plot
        self.is_print = is_print
        self.test = 12
        
        
    def fit(self,df):
        """
        df: Accept pandas dataframe
        """
        self.n_rows = df.shape[0]
        self._data = np.array(df)
        self._responsibility = np.zeros((self.n_rows,self._K))
        
        random_idx = np.random.randint(self.n_rows, size=self._K)
        self._mu = np.array(df)[random_idx,:]
        print(self._mu)
        
    def euclidean_distance(self,a,b):
        return np.linalg.norm(a-b,axis=1)
    
    def likelihood(self,index):
        """Calculates the distortion function"""
        distortion_sum = 0
        for i in range(self._K):
            index_array = np.where(self._responsibility[:,i]==1)
            distortion_sum+= np.sum(self.euclidean_distance(self._data[index_array,:],self._mu[i]))
        print(f"Distortion Sum of iteration {index}: {distortion_sum:.2f}")
        return distortion_sum

    def assignment(self):
        """Assign each data to a cluster"""
        empty_array = np.zeros_like(self._responsibility)
        for i in range(self._K):
            self._responsibility[:,i] = self.euclidean_distance(self._data,self._mu[i])
        empty_array[np.arange(self.n_rows),self._responsibility.argmin(1)]=1
        self._responsibility = empty_array

        
    def update_mu(self):
        """Update centroids of each cluster"""
        for i in range(self._K):
            index_array = np.where(self._responsibility[:,i]==1)
            self._mu[i] = np.mean(self._data[index_array,:],axis=1) 
        if self.is_print:
            print(self._mu)
        
    def optimize(self,iteration=50):
        prev_loss =0
        epsilon = 0.01
        loss_list =[]
        for i in range(iteration):
            self.assignment()
            self.update_mu()
            loss = self.likelihood(i)
            loss_list.append(loss)
            if abs(loss - prev_loss) > epsilon:
                prev_loss = loss
            else:
                print("Loss has converged")
                break
        if self.plot:
            self.draw_plot()
        plt.plot(loss_list)
    def draw_plot(self):
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        # Color Setting
        colors = iter(cm.rainbow(np.linspace(0, 1, self._K)))
        for i in range(self._K):
            index_array = np.where(self._responsibility[:,i]==1)
            color = next(colors)
            plt.plot(self._data[index_array,0],self._data[index_array,1], 'o',c=color)
            plt.plot(self._mu[i,0],self._mu[i,1],'k*',ms=14)
        plt.axis('equal')
        plt.show()

    def reconstruct_image(self,original_image_shape,save_dir=None):
        """Used to test for real world images"""
        self.reconstruct_img = np.zeros_like(self._data)
        for column in range(self._K):
            index_array = np.where(self._responsibility[:,column]==1)
            self.reconstruct_img[index_array] = self._mu[column]
        plt.imshow(self.reconstruct_img.reshape(original_image_shape))
        if save_dir is not None:
            plt.savefig(f'../data/outputs/{save_dir}',dpi=300)
            print(f'figure saved in "../data/outputs/{save_dir}"')

class SoftKMeans(KMeans):
    """Inherit the KMeans algorithm""" 
    
    def assignment(self):
        empty_array = np.zeros_like(self._responsibility)
        for i in range(self._K):
            self._responsibility[:,i] = self.euclidean_distance(self._data,self._mu[i])
        self._responsibility= self.softmax(-self.beta*self._responsibility)
        
    def update_mu(self):
        for i in range(self._K):
            numerator= np.sum(np.multiply((self._data),self._responsibility[:,i][:,np.newaxis]),axis=0)
            denominator = np.sum(self._responsibility[:,i])
            self._mu[i] = np.divide(numerator,denominator)

    def softmax(self,x):
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1,keepdims=True)
    
    def likelihood(self,index):
        distortion_sum = 0
        for i in range(self._K):
            distortion_sum+= np.sum(np.multiply(self.euclidean_distance(self._data,self._mu[i]),self._responsibility[:,i]),axis=0)    
        distortion_sum-= np.sum((1/self.beta) *np.multiply(np.log(1/self._responsibility),self._responsibility)) 
        print(f"Distortion Sum of iteration {index}: {distortion_sum:.2f}")
        return distortion_sum
        
    def optimize(self,beta=20,iteration=50):
        assert beta!=0 
        self.beta= beta
        prev_loss =0
        epsilon = 0.001
        loss_list =[]
        for i in range(iteration):
            self.assignment()
            self.update_mu()
            loss = self.likelihood(i)
            loss_list.append(loss)

            if abs(loss - prev_loss) > epsilon:
                prev_loss = loss
            else:
                print("Loss has converged")
                break
        if self.plot:
            self.draw_plot(i)
        plt.plot(loss_list)
    def draw_plot(self,index):
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        # Fix color issue
        if index%2==0:
            self._responsibility= np.flip(self._responsibility, axis=1) 
        plt.scatter(self._data[:,0],self._data[:,1],c=self._responsibility[:,0])
        plt.scatter(self._mu[:,0],self._mu[:,1],c='r',s=40)
        plt.axis('equal')
        plt.show()

    def reconstruct_image(self,original_image_shape,save_dir=None):
        self.reconstruct_img =np.dot(self._responsibility,self._mu)
        plt.imshow(self.reconstruct_img.reshape(original_image_shape))
        if save_dir is not None:
            plt.savefig(f'../data/outputs/{save_dir}',dpi=300)
            print(f'figure saved in "../data/outputs/{save_dir}"')