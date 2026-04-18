import numpy as np
import pandas as pd


def build_metric_summary_df(
    metrics_label_list,
    point_predict_all,
    interval_predict_all,
    station_name_list=None,
    method_choose_list1=None,
    interval_reduce_func=np.mean
):
    """
    Build the final summary DataFrame for point prediction and interval prediction metrics.

    Parameters
    ----------
    metrics_label_list : list[str]
        Full metric names.
        Example:
        ['MAE', 'NRMSE', 'R2', 'MAPE', 'RMSE',
         'PICP', 'PINAW', 'CWC', 'QL', 'IS']

    point_predict_all : list
        Shape: [n_methods][n_stations][n_point_metrics]

    interval_predict_all : list
        Shape: [n_methods][n_stations][n_interval_metrics, n_conf_levels]

    station_name_list : list[str], optional
        Station names used as row index.
        If None, default names will be generated automatically.

    method_choose_list1 : list[str], optional
        Method names used as column labels.
        If None, default names will be generated automatically.

    interval_reduce_func : callable, default=np.mean
        Function used to reduce interval metrics across confidence levels.

    Returns
    -------
    result_df : pd.DataFrame
        Final metric summary table with MultiIndex columns:
        Level 0 -> Metric
        Level 1 -> Method
    """

    point_arr = np.asarray(point_predict_all, dtype=float)
    interval_arr = np.asarray(interval_predict_all, dtype=float)

    if point_arr.size == 0:
        raise ValueError("point_predict_all is empty.")
    if interval_arr.size == 0:
        raise ValueError("interval_predict_all is empty.")

    if point_arr.ndim != 3:
        raise ValueError(
            f"point_predict_all should have shape [n_methods, n_stations, n_point_metrics], "
            f"but got shape={point_arr.shape}"
        )

    if interval_arr.ndim != 4:
        raise ValueError(
            f"interval_predict_all should have shape [n_methods, n_stations, n_interval_metrics, n_levels], "
            f"but got shape={interval_arr.shape}"
        )

    n_methods_p, n_stations_p, n_point_metrics = point_arr.shape
    n_methods_i, n_stations_i, n_interval_metrics, _ = interval_arr.shape

    if n_methods_p != n_methods_i or n_stations_p != n_stations_i:
        raise ValueError(
            "point_predict_all and interval_predict_all have inconsistent shapes: "
            f"{point_arr.shape} vs {interval_arr.shape}"
        )

    n_methods = n_methods_p
    n_stations = n_stations_p

    expected_metric_num = n_point_metrics + n_interval_metrics
    if len(metrics_label_list) != expected_metric_num:
        raise ValueError(
            f"metrics_label_list length should be {expected_metric_num}, "
            f"but got {len(metrics_label_list)}"
        )

    point_metric_names = metrics_label_list[:n_point_metrics]
    interval_metric_names = metrics_label_list[n_point_metrics:]

    if station_name_list is None:
        station_name_list = [f"station_{i}" for i in range(n_stations)]
    if method_choose_list1 is None:
        method_choose_list1 = [f"method_{i}" for i in range(n_methods)]

    if len(station_name_list) != n_stations:
        raise ValueError(
            f"station_name_list length should be {n_stations}, but got {len(station_name_list)}"
        )

    if len(method_choose_list1) != n_methods:
        raise ValueError(
            f"method_choose_list1 length should be {n_methods}, but got {len(method_choose_list1)}"
        )

    # -----------------------------
    # Aggregate point prediction metrics
    # point_arr: [n_methods, n_stations, n_point_metrics]
    # -> transpose to [n_stations, n_methods, n_point_metrics]
    # -----------------------------
    point_arr = np.transpose(point_arr, (1, 0, 2))

    point_df_list = []
    for k in range(n_point_metrics):
        df_k = pd.DataFrame(
            point_arr[:, :, k],
            index=station_name_list,
            columns=method_choose_list1
        )
        point_df_list.append(df_k)

    result_point_df = pd.concat(point_df_list, axis=1)

    # -----------------------------
    # Aggregate interval prediction metrics
    # interval_arr: [n_methods, n_stations, n_interval_metrics, n_levels]
    # First reduce the last axis (confidence levels), then transpose
    # -----------------------------
    interval_arr_reduced = np.apply_along_axis(interval_reduce_func, -1, interval_arr)
    interval_arr_reduced = np.transpose(interval_arr_reduced, (1, 0, 2))

    interval_df_list = []
    for k in range(n_interval_metrics):
        df_k = pd.DataFrame(
            interval_arr_reduced[:, :, k],
            index=station_name_list,
            columns=method_choose_list1
        )
        interval_df_list.append(df_k)

    result_interval_df = pd.concat(interval_df_list, axis=1)

    # -----------------------------
    # Combine point and interval metrics
    # -----------------------------
    result_df = pd.concat([result_point_df, result_interval_df], axis=1)

    # -----------------------------
    # Build two-level column index
    # Level 0: Metric
    # Level 1: Method
    # -----------------------------
    all_metric_names = point_metric_names + interval_metric_names
    multi_cols = pd.MultiIndex.from_tuples(
        [(metric, method) for metric in all_metric_names for method in method_choose_list1],
        names=['Metric', 'Method']
    )

    result_df.columns = multi_cols

    return result_df


