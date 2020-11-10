import numpy as np


class Kmeans:

    def __init__(self,no_of_clusters,max_itr=100,random_state=100):
        self.no_of_clusters=no_of_clusters
        self.random_state=random_state
        self.max_itr=max_itr
    
    def euclid_distance(self,p1,p2):
        """ Distance function
        Calculates the euclidean distance between the test_point and the input X point

        :Parameters:
        p1 : test point
        p2 : Input X point

        :Return:
        The euclidean distance between the test_point and the input X point

        """
        distance= np.sum((p1-p2)**2)**0.5
        return distance
    
    def initialize_centroids(self,X):
    
        np.random.RandomState(self.random_state)
        random_index=np.random.permutation(X.shape[0])
        centroids= X[random_index[:self.no_of_clusters]]
        return centroids
    
    def find_distance(self,X,centroids):
        distance=np.zeros((X.shape[0],self.no_of_clusters))
        for k in range(self.no_of_clusters):
            norm=np.linalg.norm(X-centroids[k,:],axis=1)
            distance[:,k]=np.square(norm)
        return distance

    def find_nearest(self,distance):
        return np.argmin(distance,axis=1)
    
    


    


     

    

            