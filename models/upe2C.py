import numpy as np
from pymoo.optimize import minimize
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from models.moo_solver1 import MyProblem1


class UPE2CModel:
    def __init__(self,tabpfn, tabm, zhixin) -> None:

        self.tabpfn = tabpfn
        self.tabm = tabm
        self.zhixin = zhixin
        self.optimized_W = None

    def _normalize_weights(self, w, min_weight=0.25):
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
    

    def train(
        self,
        x_aug: np.ndarray,
        y: np.ndarray,
        val_aug_x_nor: np.ndarray,
        val_y_nor: np.ndarray,
        y_vaild: np.ndarray,
        y_mean,
        y_std
    ) -> None:
        # Train base models
        # self.ngb.train(x_aug, y, val_aug_x_nor, val_y_nor)
        # y_middle_vaild_ngb_aug, upper_vaild_ngb_aug, lower_vaild_ngb_aug = \
        #     self.ngb.predict(val_aug_x_nor, y_mean, y_std)

        self.tabpfn.train(x_aug, y, val_aug_x_nor, val_y_nor)
        y_middle_vaild_tabpfn, upper_vaild_tabpfn, lower_vaild_tabpfn = \
            self.tabpfn.predict(val_aug_x_nor, y_mean, y_std)

        self.tabm.train(x_aug, y, val_aug_x_nor, val_y_nor)
        y_middle_vaild_tabm_aug, upper_vaild_tabm_aug, lower_vaild_tabm_aug = \
            self.tabm.predict(val_aug_x_nor, y_mean, y_std)

        # Multi-objective optimizer
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=50,
            prob_neighbor_mating=0.7
        )

        # Use interval confidence levels except the first one
        zhixin1 = self.zhixin[1:]

        pre_vaild_y_list_all_optimize1 = [
            y_middle_vaild_tabpfn,
            y_middle_vaild_tabm_aug
        ]

        pre_vaild_y_list_upper_all_optimize1 = [
            upper_vaild_tabpfn,
            upper_vaild_tabm_aug
        ]

        pre_vaild_y_list_lower_all_optimize1 = [
            lower_vaild_tabpfn,
            lower_vaild_tabm_aug
        ]

        # Debug information
        # print("y_valid mean:", np.mean(y_vaild), "std:", np.std(y_vaild))

        # for k, arr in enumerate(pre_vaild_y_list_all_optimize1):
        #     print(f"middle model {k}: mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")

        # for k, arr in enumerate(pre_vaild_y_list_upper_all_optimize1):
        #     print(f"upper model {k}: mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")

        # for k, arr in enumerate(pre_vaild_y_list_lower_all_optimize1):
        #     print(f"lower model {k}: mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")

        # Run optimization
        method_optimize=2
        problem = MyProblem1(
            zhixin1,
            y_vaild,
            pre_vaild_y_list_all_optimize1,
            pre_vaild_y_list_lower_all_optimize1,
            pre_vaild_y_list_upper_all_optimize1,method_optimize
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', 100),
            seed=1,
            verbose=False
        )

        print("res.X shape:", res.X.shape)
        print("res.F shape:", res.F.shape)

        F = res.F.copy()

        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_norm = (F - F_min) / (F_max - F_min + 1e-12)

        w1, w2, w3 = 0.3, 0.3, 0.4
        score = w1 * F_norm[:, 0] + w2 * F_norm[:, 1] + w3 * F_norm[:, 2]

        best_idx = np.argmin(score)
        self.optimized_W = res.X[best_idx].copy()

        # print("best_idx:", best_idx)
        # print("best objectives:", res.F[best_idx])
        # print("best weights vector shape:", self.optimized_W.shape)
        # print("best weights vector:", self.optimized_W)

    def predict(self, x_aug: np.ndarray, y_mean: float, y_std: float):
        if self.optimized_W is None:
            raise ValueError("The model has not been trained yet. Please call train() first.")

        # Predict from base models
        # ngb_middle, ngb_upper, ngb_lower = self.ngb.predict(x_aug, y_mean, y_std)
        tabpfn_middle, tabpfn_upper, tabpfn_lower = self.tabpfn.predict(x_aug, y_mean, y_std)
        tabm_middle, tabm_upper, tabm_lower = self.tabm.predict(x_aug, y_mean, y_std)

        middles = [ tabpfn_middle, tabm_middle]
        uppers = [ tabpfn_upper, tabm_upper]
        lowers = [ tabpfn_lower, tabm_lower]

        # Reshape one selected solution:
        # shape = (n_models, n_interval_columns + 1 point column)
        AA1 = self.optimized_W.reshape(2, len(self.zhixin))

        predict_length = lowers[0].shape[0]
        n_interval_cols = AA1.shape[1] - 1

        lower_last_test = np.zeros((predict_length, n_interval_cols))
        upper_last_test = np.zeros((predict_length, n_interval_cols))
        mean_last_test = np.zeros_like(middles[0], dtype=float)

        # Fuse interval predictions
        for i in range(n_interval_cols):
            W1 = AA1[:, i]
            W2 = self._normalize_weights(W1)

            print(f"Interval column {i}, normalized weights: {W2}")

            for j in range(2):
                lower_last_test[:, i] += W2[j] * lowers[j][:, i]
                upper_last_test[:, i] += W2[j] * uppers[j][:, i]

        # Fuse point prediction using the last column
        W1 = AA1[:, -1]
        W2 = self._normalize_weights(W1)

        print(f"Point prediction normalized weights: {W2}")

        for j in range(2):
            mean_last_test += W2[j] * middles[j]

        return mean_last_test, upper_last_test, lower_last_test