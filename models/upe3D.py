import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pymoo.optimize import minimize
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from models.moo_solver import MyProblem1


class UPE3AModel:
    def __init__(self, ngb, tabpfn, tabm, zhixin) -> None:
        self.ngb = ngb
        self.tabpfn = tabpfn
        self.tabm = tabm
        self.zhixin = zhixin
        self.optimized_W = None

    def _normalize_weights(self, w, min_weight=0.15):
        w = np.asarray(w, dtype=float)
        n = len(w)

        if min_weight * n >= 1.0:
            raise ValueError(
                f"min_weight={min_weight} is too large for {n} models. "
                f"It must satisfy min_weight * n < 1."
            )

        s = np.sum(w)

        if s <= 1e-12:
            base = np.ones(n) / n
        else:
            base = w / s

        weights = min_weight + (1.0 - n * min_weight) * base
        return weights

    def _train_and_predict_one_model(
        self,
        model,
        x_aug,
        y,
        val_aug_x_nor,
        val_y_nor,
        y_mean,
        y_std
    ):

        model.train(x_aug, y, val_aug_x_nor, val_y_nor)
        y_middle, upper, lower = model.predict(val_aug_x_nor, y_mean, y_std)
        return y_middle, upper, lower

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

        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_ngb = executor.submit(
                self._train_and_predict_one_model,
                self.ngb, x_aug, y, val_aug_x_nor, val_y_nor, y_mean, y_std
            )
            future_tabpfn = executor.submit(
                self._train_and_predict_one_model,
                self.tabpfn, x_aug, y, val_aug_x_nor, val_y_nor, y_mean, y_std
            )
            future_tabm = executor.submit(
                self._train_and_predict_one_model,
                self.tabm, x_aug, y, val_aug_x_nor, val_y_nor, y_mean, y_std
            )

            y_middle_vaild_ngb_aug, upper_vaild_ngb_aug, lower_vaild_ngb_aug = future_ngb.result()
            y_middle_vaild_tabpfn, upper_vaild_tabpfn, lower_vaild_tabpfn = future_tabpfn.result()
            y_middle_vaild_tabm_aug, upper_vaild_tabm_aug, lower_vaild_tabm_aug = future_tabm.result()

       
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=50,
            prob_neighbor_mating=0.7
        )

        zhixin1 = self.zhixin[1:]

        pre_vaild_y_list_all_optimize1 = [
            y_middle_vaild_ngb_aug,
            y_middle_vaild_tabpfn,
            y_middle_vaild_tabm_aug
        ]

        pre_vaild_y_list_upper_all_optimize1 = [
            upper_vaild_ngb_aug,
            upper_vaild_tabpfn,
            upper_vaild_tabm_aug
        ]

        pre_vaild_y_list_lower_all_optimize1 = [
            lower_vaild_ngb_aug,
            lower_vaild_tabpfn,
            lower_vaild_tabm_aug
        ]

        method_optimize = 3
        problem = MyProblem1(
            zhixin1,
            y_vaild,
            pre_vaild_y_list_all_optimize1,
            pre_vaild_y_list_lower_all_optimize1,
            pre_vaild_y_list_upper_all_optimize1,
            method_optimize
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

        w1, w2, w3 = 0.4, 0.3, 0.3
        score = w1 * F_norm[:, 0] + w2 * F_norm[:, 1] + w3 * F_norm[:, 2]

        best_idx = np.argmin(score)
        self.optimized_W = res.X[best_idx].copy()

    def predict(self, x_aug: np.ndarray, y_mean: float, y_std: float):
        if self.optimized_W is None:
            raise ValueError("The model has not been trained yet. Please call train() first.")

        ngb_middle, ngb_upper, ngb_lower = self.ngb.predict(x_aug, y_mean, y_std)
        tabpfn_middle, tabpfn_upper, tabpfn_lower = self.tabpfn.predict(x_aug, y_mean, y_std)
        tabm_middle, tabm_upper, tabm_lower = self.tabm.predict(x_aug, y_mean, y_std)

        middles = [ngb_middle, tabpfn_middle, tabm_middle]
        uppers = [ngb_upper, tabpfn_upper, tabm_upper]
        lowers = [ngb_lower, tabpfn_lower, tabm_lower]

        AA1 = self.optimized_W.reshape(3, len(self.zhixin))

        predict_length = lowers[0].shape[0]
        n_interval_cols = AA1.shape[1] - 1

        lower_last_test = np.zeros((predict_length, n_interval_cols))
        upper_last_test = np.zeros((predict_length, n_interval_cols))
        mean_last_test = np.zeros_like(middles[0], dtype=float)

        for i in range(n_interval_cols):
            W1 = AA1[:, i]
            W2 = self._normalize_weights(W1)

            print(f"Interval column {i}, normalized weights: {W2}")

            for j in range(3):
                lower_last_test[:, i] += W2[j] * lowers[j][:, i]
                upper_last_test[:, i] += W2[j] * uppers[j][:, i]

        W1 = AA1[:, -1]
        W2 = self._normalize_weights(W1)

        print(f"Point prediction normalized weights: {W2}")

        for j in range(3):
            mean_last_test += W2[j] * middles[j]

        return mean_last_test, upper_last_test, lower_last_test