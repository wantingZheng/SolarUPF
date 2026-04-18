import numpy as np
import lightgbm as lgb

from typing import List


class CustomLightgbm:
    def __init__(self, zhixin: List=[0,0.3,0.5,0.7,0.9]) -> None:
        quantiles = 1 - (1 - np.array(zhixin)) / 2
        self.quantiles = quantiles
        self.models = []
        self.zhixin=zhixin
        
    def train(self, x, y, x_val, y_val, num_boost_round: int = 100) -> List:

        train_data = lgb.Dataset(x, y)
        valid_data = lgb.Dataset(x_val, y_val, reference=train_data)

        self.models = []

        early_stop_rounds = 20

        for q in self.quantiles:

            # -------- lower quantile --------
            # params1 = {
            #     "objective": "quantile",
            #     "alpha": q,
            #     "force_col_wise": True,
            #     "verbosity": -1
            # }
            params1 = {
            "objective": "quantile",
            "alpha": q,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "min_gain_to_split": 0.01,
            "max_bin": 255,
            "force_col_wise": True,
            "verbosity": -1,
            "seed": 42}

            model_low = lgb.train(
                params=params1,
                train_set=train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
            )

            self.models.append(model_low)
            
        
        for q in self.quantiles:
            # -------- upper quantile --------
            if q != 0.5:
                params2 = {
                "objective": "quantile",
                "alpha": 1-q,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "lambda_l2": 1.0,
                "min_gain_to_split": 0.01,
                "max_bin": 255,
                "force_col_wise": True,
                "verbosity": -1,
                "seed": 42}                
                # params2 = {
                #     "objective": "quantile",
                #     "alpha": 1 - q,
                #     "force_col_wise": True,
                #     "verbosity": -1
                # }

                model_high = lgb.train(
                    params=params2,
                    train_set=train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[train_data, valid_data],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
                )

                self.models.append(model_high)

        return self.models
    
    def predict(self, x, y_mean, y_std):
        upper = np.zeros((len(x),len(self.quantiles) - 1)) 
        lower = np.zeros((len(x),len(self.quantiles) - 1))
        for i, model in enumerate(self.models):
            y = model.predict(x)
            y = y * y_std + y_mean
            if i == 0:
                y_middle = y
            elif i <= len(self.zhixin)-1:
                upper[:, (i - 1) % len(self.quantiles)] = y
            else:
                lower[:, (i - len(self.zhixin)) % len(self.quantiles)] = y
                
        return y_middle, upper, lower