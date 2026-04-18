import numpy as np
from pymoo.core.problem import Problem
from utils.metrics1 import cacluate_interval_score, evaluate_regress

class MyProblem1(Problem):
    def __init__(self, zhixin, y_vaild,
                 pre_vaild_y_list_all_optimize1,
                 pre_vaild_y_list_lower_all_optimize1,
                 pre_vaild_y_list_upper_all_optimize1,method_optimize):

        self.true = y_vaild
        self.lower = pre_vaild_y_list_lower_all_optimize1
        self.upper = pre_vaild_y_list_upper_all_optimize1
        self.pre = pre_vaild_y_list_all_optimize1
        self.zhixin = zhixin

        # method_optimize = 3
        self.rows = method_optimize

        zhixin_length = pre_vaild_y_list_lower_all_optimize1[0].shape[1]
        predict_length = pre_vaild_y_list_lower_all_optimize1[0].shape[0]

        self.cols = zhixin_length + 1
        self.pres = predict_length

        n_var1 = method_optimize * (zhixin_length + 1)

        super().__init__(
            n_var=n_var1,
            n_obj=3,
            n_constr=0,
            xl=np.zeros(n_var1),
            xu=np.ones(n_var1)
        )

    # def _normalize_weights(self, w):
    #     s = np.sum(w)
    #     if s <= 1e-12:
    #         return np.ones_like(w) / len(w)
    #     return w / s
    def _normalize_weights( self,w, min_weight=0.25):
        """
        Normalize weights with a lower bound constraint.

        Parameters
        ----------
        w : array-like
            Raw weights.
        min_weight : float
            Minimum weight assigned to each model.

        Returns
        -------
        np.ndarray
            Normalized weights with each element >= min_weight
            and total sum equal to 1.
        """
        w = np.asarray(w, dtype=float)
        n = len(w)

        # Check feasibility of the minimum weight constraint
        if min_weight * n >= 1.0:
            raise ValueError(
                f"min_weight={min_weight} is too large for {n} models. "
                f"It must satisfy min_weight * n < 1."
            )

        s = np.sum(w)

        # If all raw weights are nearly zero, assign uniform weights first
        if s <= 1e-12:
            base = np.ones(n) / n
        else:
            base = w / s

        # Map to lower-bounded simplex
        weights = min_weight + (1.0 - n * min_weight) * base

        return weights
    
    def _evaluate(self, X, out, *args, **kwargs):

        pre = self.pre
        true = self.true
        lower = self.lower
        upper = self.upper
        pres = self.pres

        matrices = X.reshape(-1, self.rows, self.cols)

        n_solutions = matrices.shape[0]

        f1 = np.zeros(n_solutions)
        f2 = np.zeros(n_solutions)
        f3 = np.zeros(n_solutions)

        for N1 in range(n_solutions):

            lower_last = np.zeros((pres, self.cols - 1))
            upper_last = np.zeros((pres, self.cols - 1))
            predict_last = np.zeros_like(true, dtype=float)

            # Interval prediction fusion
            for i in range(self.cols - 1):
                W1 = matrices[N1, :, i]
                W2 = self._normalize_weights(W1)

                for j in range(self.rows):
                    lower_last[:, i] += W2[j] * lower[j][:, i]
                    upper_last[:, i] += W2[j] * upper[j][:, i]

            # Point prediction fusion
            W1 = matrices[N1, :, self.cols - 1]
            W2 = self._normalize_weights(W1)

            for j in range(self.rows):
                predict_last += W2[j] * pre[j]

            zhixin = self.zhixin
            cacluate_interval = np.zeros((5, len(zhixin)))

            for i in range(self.cols - 1):
                confidence_level = zhixin[i]
                alpha = 1 - confidence_level

                lower_bounds = lower_last[:, i]
                upper_bounds = upper_last[:, i]

                picp_result, pinaw_result, cwc_result, interval_score_result,ql_result,  = \
                    cacluate_interval_score(true, lower_bounds, upper_bounds, 10, alpha)

                cacluate_interval[:, i] = [
                    picp_result,
                    pinaw_result,
                    cwc_result,
                    interval_score_result,
                    ql_result
                ]

            MAE, NRMSE, R2, MAPE, RMSE = evaluate_regress(predict_last, true)

            f1[N1] = np.mean(cacluate_interval[2, :])   # mean CWC
            f2[N1] = np.mean(cacluate_interval[3, :])   # mean IS
            f3[N1] = np.mean(NRMSE)                     # NRMSE

        out["F"] = np.column_stack([f1, f2, f3])