import numpy as np

class LogisticRegression:
      
    
    def normalize(self,X):
        mean=X.mean(axis=0)
        
        standard_deviation=X.std(axis=0)
        
        X_normalized=(X-mean)/standard_deviation
        self.X=X_normalized
        return X_normalized
        
    def add_ones(self,X):
        ones = np.ones((X.shape[0],1))
        X_new= np.hstack((ones,X))
        self.X=X_new
        return X_new

    def sigmoid(self,z):
        return 1/(1.0+np.exp(-z))

    def hypothesis(self,X,theta):
        """ Hypothesis Function
        Predicts the output y based on the input x and model parameters

        :Parameters:
        X  : Depentent variables which the response variable to be calculated
        theta : Model parameters

        :Return:
        Prediction values

        """
        y=self.sigmoid(np.dot(theta,X))
        return y
    
    def cost_function(self,X,Y,theta):
        """ Cost Function
        It returns the cost(error)

        :Parameters:
        X  : Depentent variables
        Y  : Corresponding response to X 
        theta : Model parameters

        :Return:
        Calculated Cost

        """

        size=X.shape[0]
        Y_pred=self.hypothesis(theta,X)
        
        cost= -(np.sum(Y * np.log (Y_pred)  + (1-Y) * np.log (1-Y_pred)))
        # print(Y_pred)
        return cost*(1/size)
    
    def gradient(self,X,Y,theta):
        """ Gradient Function
        It returns the gradient value

        :Parameters:
        X  : Depentent variables
        Y  : Corresponding response to X 
        theta : Model parameters

        :Return:
        Calculated gradient

        """
       
        Y_pred=self.hypothesis(theta,X)
        gradient=np.dot(X.T,(Y_pred-Y))
        return gradient
    
    def gradient_descent(self,X,Y,lr=0.0001,iteration=200):
        """ Gradient Function
        It returns the gradient value

        :Parameters:
        lr : Learning rate (default=0.1)
        iteration : No of iteration to run (default=500)

        :Return:
        List of costs at different theta throughout the training

        """
        
        theta = np.zeros((X.shape[1],1))
        cost_list=[]
        for i in range(iteration):
            grad=self.gradient(X,Y,theta)
            cost=self.cost_function(X,Y,theta)
            theta=theta - lr*grad
            cost_list.append(cost)

        return theta,cost_list
    
    def predict(self,theta,X):
        
        Y_pred = self.hypothesis(theta,X)
        
        return Y_pred

