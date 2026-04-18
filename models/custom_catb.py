import numpy as np
from catboost import CatBoostRegressor
from typing import List

class CustomCatBoost:
    def __init__(self, zhixin: List=[0,0.3,0.5,0.7,0.9]) -> None:
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.models = []
        self.zhixin=zhixin
        
    def train(self, x, y,x_val,y_val, depth: int=5,learning_rate=0.04,iterations: int=100) -> List:
        for q in self.quantiles:
            quantile_str1='Quantile:alpha='+str(q)
            if q == 0.5:
                model = CatBoostRegressor(loss_function=quantile_str1,depth=depth, learning_rate=learning_rate, iterations=iterations,early_stopping_rounds=10,verbose=False)
                model.fit(x, y,eval_set=(x_val, y_val), use_best_model=True)
                self.models.append(model)
            else:
                model = CatBoostRegressor(loss_function=quantile_str1,depth=depth, learning_rate=learning_rate, iterations=iterations,early_stopping_rounds=10,verbose=False)
                model.fit(x, y,eval_set=(x_val, y_val), use_best_model=True)
                
                self.models.append(model)                
        for q in self.quantiles:
            quantile_str1='Quantile:alpha='+str(1-q)
            if q != 0.5:
                model = CatBoostRegressor(loss_function=quantile_str1,depth=depth, learning_rate=learning_rate, iterations=iterations,early_stopping_rounds=10,verbose=False)
                model.fit(x, y,eval_set=(x_val, y_val), use_best_model=True)
                self.models.append(model)  
            
        return self.models
    
    def predict(self, x, y_mean, y_std):
        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        for i, model in enumerate(self.models):
            y = model.predict(x)
            y = y * y_std + y_mean
            if i == 0:
                y_middle = y
            elif i <=len(self.zhixin)-1:
                upper[:, (i - 1) % len(self.quantiles)] = y
            else:
                lower[:, (i - len(self.zhixin)) % len(self.quantiles)] = y
                
        return y_middle, upper, lower