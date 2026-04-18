import numpy as np
from typing import List
from tabicl import TabICLRegressor

class CustomTabICL:
    def __init__(self, zhixin: List) -> None:
        # zhixin=[0, 0.3, 0.5, 0.7 ,0.9]
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.fenwei = np.append(quantiles, 1 - quantiles[1:])
        self.model = TabICLRegressor()
        self.zhixin=zhixin
        
    def train(self, x, y,x_val,y_val):
        self.model.fit(x, y)
        
    def predict(self, x, y_mean, y_std):
        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        y = self.model.predict(x, output_type="quantiles", alphas=self.fenwei)
        for i in range(len(self.fenwei)):
            if i == 0:
       
                y_middle = y[:,i] * y_std + y_mean
            elif i <= len(self.zhixin)-1:
                upper[:, (i - 1) % len(self.quantiles)] = y[:,i] * y_std + y_mean
            else:
                lower[:, (i - len(self.zhixin)) % len(self.quantiles)] = y[:,i] * y_std + y_mean
                
        return y_middle, upper, lower