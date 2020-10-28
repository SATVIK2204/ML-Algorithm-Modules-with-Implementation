import numpy as np

class KNN:

    def __init__(self):
        super().__init__()

    def euclid_distance(self,p1,p2):
        """ Distance function
        Calculates the euclidean distance between the test_point and the input X point

        :Parameters:
        p1 : test point
        p2 : Input X point

        :Return:
        The euclidean distance between the test_point and the input X point

        """
        return np.sum((p1-p2)**2)**0.5

    def model(self,X,Y,test_point,k=5):
        """ Distance function
        Calculates the euclidean distance between the test_point and the input X point

        :Parameters:
        X : Input X
        Y : Input Y labels
        test_point : Point for which label is to found

        :Return:
        The label to which test point belongs.

        """

        distance_point_pair=[]
        size=X.shape[0]
        for i in range(size):
            distance=self.euclid_distance(test_point,X[i])
            distance_point_pair.append((distance,Y[i]))

        distance_point_pair.sort() 
        top_k_distance_point_pair=np.array(distance_point_pair[:k])[:,1]
        unique,freq=np.unique(top_k_distance_point_pair,return_counts=True)
        max_freq_index=np.argmax(freq)
        label=unique[max_freq_index]

        return label


            