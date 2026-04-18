import numpy as np
import statsmodels.api as sm
from typing import List

class CustomLinear:
    def __init__(self, zhixin: List = [0, 0.3, 0.5, 0.7, 0.9]) -> None:
        """
        zhixin:  [0, 0.3, 0.5, 0.7, 0.9]
                
        """
        self.zhixin = zhixin
      
        self.quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.model = None

    def train(self, x, y, X_val=0, y_val=0):
        """
        statsmodels: OLS(endog=y, exog=X)
        """
        X = np.asarray(x)
        y = np.asarray(y).reshape(-1)  # 

    
        X = sm.add_constant(X, has_constant='add')

        self.model = sm.OLS(y, X).fit()
        return self.model

    def predict(self, x, y_mean, y_std):
        """
          y_middle: [N,]    predicted_mean）
          upper:    [N, K]  upper）
          lower:    [N, K]  lower）
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        X = np.asarray(x)
        X = sm.add_constant(X, has_constant='add')

        pred = self.model.get_prediction(X)

        mean_pred = pred.predicted_mean
        y_middle = mean_pred * y_std + y_mean

        conf_levels = [c for c in self.zhixin if c > 0]
        K = len(conf_levels)

        upper = np.zeros((len(X), K))
        lower = np.zeros((len(X), K))

        for i, conf in enumerate(conf_levels):
   
            alpha = 1.0 - conf

            sf = pred.summary_frame(alpha=alpha)
            lower_i = sf["obs_ci_lower"].to_numpy()
            upper_i = sf["obs_ci_upper"].to_numpy()

            lower[:, i] = lower_i * y_std + y_mean
            upper[:, i] = upper_i * y_std + y_mean

        return y_middle, upper, lower
