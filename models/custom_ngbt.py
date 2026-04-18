import numpy as np
from typing import List
from ngboost import NGBRegressor
from ngboost.distns import Normal

class CustomNGBoost:
    def __init__(self, zhixin: List=[0,0.3,0.5,0.7,0.9]) -> None:
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.models = []
        self.zhixin=zhixin
        
    def train(self, x, y , X_val,y_val, n_estimators: int=200,learning_rate=0.05) -> List:

        ngb = NGBRegressor(
            Dist=Normal,
            n_estimators=n_estimators,             
            learning_rate=learning_rate,
            early_stopping_rounds=10,

        )
        # ngb = NGBRegressor(Dist=Normal)
        # train
        ngb.fit(x, y, X_val=X_val,Y_val=y_val)
        self.models=ngb
            
        return self.models
    
    def predict(self, x, y_mean, y_std):
        preds = self.models.pred_dist(x)
        
        means = preds.params['loc']
        stddevs = preds.params['scale']

        y_middle = means * y_std + y_mean

        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        for i in range(len(self.quantiles) - 1):
            # confidence_level = 0.95
            confidence_level=self.zhixin[i+1]

            z_score = np.abs(np.percentile(np.random.normal(size=10000), (1 - confidence_level) * 100 / 2))
            lower_bounds = means - z_score * stddevs
            upper_bounds = means + z_score * stddevs
            upper[:,i]=upper_bounds*y_std+y_mean
            lower[:,i]=lower_bounds*y_std+y_mean
        

        return y_middle, upper, lower