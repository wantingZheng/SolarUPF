from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

def safe_std(arr: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    std = np.std(arr, axis=axis, keepdims=keepdims)
    std = np.where(std == 0, 1.0, std)
    return std


def flatten_pred_matrix(y_middle_matrix: np.ndarray,
                        upper_matrix: np.ndarray,
                        lower_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [n_window, 96] / [n_window, 96, n_q] 
    y_middle -> [n_window*96]
    upper/lower -> [n_window*96, n_q]
    """
    y_middle = y_middle_matrix.reshape(-1)
    upper = upper_matrix.reshape(-1, upper_matrix.shape[-1])
    lower = lower_matrix.reshape(-1, lower_matrix.shape[-1])
    return y_middle, upper, lower


def build_nonoverlap_target(df: pd.DataFrame,
                            target_col: str,
                            start_idx: int,
                            end_idx: int) -> np.ndarray:
    return df.iloc[start_idx:end_idx][target_col].to_numpy(dtype=np.float32)


def build_seq_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    past_lags: int,
    horizon: int,
    origin_start: int,
    origin_end: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - X_past   : [n, past_lags, n_features + 1]  
    - X_future : [n, horizon, n_features]       
    - Y        : [n, horizon]                    
    - origins  : [n]
    """
    df = df.reset_index(drop=True)
    past_feature_cols = list(feature_cols) + [target_col]

    x_past_list = []
    x_future_list = []
    y_list = []
    origins = []

    # origin=t [t-past_lags, ..., t-1][t, ..., t+horizon-1]
    start = max(origin_start, past_lags)
    stop = origin_end - horizon + 1

    if stop <= start:
        return (
            np.empty((0, past_lags, len(past_feature_cols)), dtype=np.float32),
            np.empty((0, horizon, len(feature_cols)), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            np.empty((0,), dtype=np.int32)
        )

    for t in range(start, stop, stride):
        past_block = df.iloc[t-past_lags:t][past_feature_cols].to_numpy(dtype=np.float32)
        future_known = df.iloc[t:t+horizon][feature_cols].to_numpy(dtype=np.float32)
        future_y = df.iloc[t:t+horizon][target_col].to_numpy(dtype=np.float32)

        x_past_list.append(past_block)
        x_future_list.append(future_known)
        y_list.append(future_y)
        origins.append(t)

    return (
        np.asarray(x_past_list, dtype=np.float32),
        np.asarray(x_future_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(origins, dtype=np.int32)
    )


def fit_seq_norm(
    x_past_train: np.ndarray,
    x_future_train: np.ndarray,
    y_train: np.ndarray
) -> Dict[str, np.ndarray]:
    norm_dict = {
        'x_past_mean': np.mean(x_past_train, axis=(0, 1), keepdims=True),
        'x_past_std': safe_std(x_past_train, axis=(0, 1), keepdims=True),
        'x_future_mean': np.mean(x_future_train, axis=(0, 1), keepdims=True),
        'x_future_std': safe_std(x_future_train, axis=(0, 1), keepdims=True),
        'y_mean': np.mean(y_train),
        'y_std': float(np.std(y_train)) if float(np.std(y_train)) != 0 else 1.0
    }
    return norm_dict


def apply_seq_norm(
    x_past: np.ndarray,
    x_future: np.ndarray,
    y: Optional[np.ndarray],
    norm_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x_past_nor = (x_past - norm_dict['x_past_mean']) / norm_dict['x_past_std']
    x_future_nor = (x_future - norm_dict['x_future_mean']) / norm_dict['x_future_std']

    if y is None:
        y_nor = None
    else:
        y_nor = (y - norm_dict['y_mean']) / norm_dict['y_std']

    return x_past_nor.astype(np.float32), x_future_nor.astype(np.float32), None if y_nor is None else y_nor.astype(np.float32)
