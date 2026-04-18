import numpy as np
from typing import List
from sklearn.linear_model import BayesianRidge

class CustomBaye:
    def __init__(self, zhixin: List=[0,0.3,0.5,0.7,0.9]) -> None:
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.models = []
        self.zhixin=zhixin
        
    def train(self, x, y , X_val=0,y_val=0) -> List:
        model = BayesianRidge()
        model.fit(x, y)
        self.models=model
            
        return self.models
    
    def predict(self, x, y_mean, y_std):
        
        means, stddevs = self.models.predict(x, return_std=True)

        y_middle = means * y_std + y_mean

        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        for i in range(len(self.quantiles) - 1):
            confidence_level=self.zhixin[i+1]

            z_score = np.abs(np.percentile(np.random.normal(size=10000), (1 - confidence_level) * 100 / 2))
            lower_bounds = means - z_score * stddevs
            upper_bounds = means + z_score * stddevs
            upper[:,i]=upper_bounds*y_std+y_mean
            lower[:,i]=lower_bounds*y_std+y_mean
        

        return y_middle, upper, lower