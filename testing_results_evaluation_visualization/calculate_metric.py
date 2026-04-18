import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import cm
import matplotlib.pyplot as plt 
def compute_normalized_regret(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute site-wise normalized regret:
        r_{s,j} = (z_{s,j} - min_l z_{s,l}) / (max_l z_{s,l} - min_l z_{s,l})

    Parameters
    ----------
    data : pd.DataFrame
        rows = sites, columns = algorithms/models
        values = evaluation metric z_{s,j}, smaller is better

    Returns
    -------
    pd.DataFrame
        same shape as data, containing normalized regrets
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    z = data.to_numpy(dtype=float)
    z_min = z.min(axis=1, keepdims=True)
    z_max = z.max(axis=1, keepdims=True)
    denom = z_max - z_min

    # When all algorithms have the same score on one site, set regret row to 0
    r = np.divide(
        z - z_min,
        denom,
        out=np.zeros_like(z, dtype=float),
        where=denom > 0
    )

    return pd.DataFrame(r, index=data.index, columns=data.columns)


def _coerce_capacity_series(capacities, site_index):
    """
    Convert capacities to a pd.Series aligned with site_index.
    """
    if isinstance(capacities, pd.Series):
        cap = capacities.reindex(site_index)
        if cap.isna().any():
            missing = cap[cap.isna()].index.tolist()
            raise ValueError(f"Missing capacities for sites: {missing}")
        cap = cap.astype(float)
    else:
        cap = np.asarray(capacities, dtype=float).reshape(-1)
        if len(cap) != len(site_index):
            raise ValueError(
                f"capacities length ({len(cap)}) must equal number of sites ({len(site_index)})"
            )
        cap = pd.Series(cap, index=site_index, dtype=float)

    if (cap < 0).any():
        raise ValueError("capacities must be non-negative")

    return cap


def calculate_opt_term(data: pd.DataFrame):
    """
    First aggregation term: optimized site-weighted term A_opt.

    This implementation follows the logic of your current code:
    - compute normalized regret matrix
    - derive site weights from the transformed matrix
    - aggregate regrets using optimized site weights

    Returns
    -------
    A_opt : pd.Series
        aggregated score for each algorithm
    w_opt : pd.Series
        optimized site weights
    r_df : pd.DataFrame
        normalized regret matrix
    """
    r_df = compute_normalized_regret(data)

    # W: rows = algorithms, cols = sites
    W = r_df.to_numpy().T
    gram = W @ W.T
    diag = np.diag(gram).reshape(-1, 1)

    try:
        a = np.linalg.solve(gram, diag)
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(gram) @ diag

    a = a.ravel()

    # Keep consistent with your original logic
    if np.min(a) < 0:
        a = a - np.min(a)

    if np.isclose(a.sum(), 0):
        a = np.ones_like(a) / len(a)
    else:
        a = a / a.sum()

    # Derived site weights
    site_weights = np.asarray(a @ W).ravel()

    # Enforce nonnegative weights for stability
    site_weights = np.clip(site_weights, 0, None)

    if np.isclose(site_weights.sum(), 0):
        site_weights = np.ones(r_df.shape[0]) / r_df.shape[0]
    else:
        site_weights = site_weights / site_weights.sum()

    # A_opt for each algorithm
    A_opt = r_df.to_numpy().T @ site_weights

    return (
        pd.Series(A_opt, index=data.columns, name="A_opt"),
        pd.Series(site_weights, index=data.index, name="w_opt"),
        r_df,
    )


def calculate_capacity_term(data: pd.DataFrame, capacities, r_df: pd.DataFrame = None):
    """
    Second aggregation term: capacity-weighted term A_cap.

    Formula:
        A_j^cap = sum_s r_{s,j} * ln(1 + C_s) / sum_u ln(1 + C_u)

    Returns
    -------
    A_cap : pd.Series
        capacity-weighted score for each algorithm
    w_cap : pd.Series
        capacity weights for each site
    """
    if r_df is None:
        r_df = compute_normalized_regret(data)

    cap = _coerce_capacity_series(capacities, r_df.index)

    w_cap = np.log1p(cap)
    if np.isclose(w_cap.sum(), 0):
        w_cap[:] = 1.0 / len(w_cap)
    else:
        w_cap = w_cap / w_cap.sum()

    A_cap = r_df.to_numpy().T @ w_cap.to_numpy()

    return (
        pd.Series(A_cap, index=r_df.columns, name="A_cap"),
        pd.Series(w_cap.to_numpy(), index=r_df.index, name="w_cap"),
    )


def _empirical_cvar(losses, beta=0.9):
    """
    Empirical CVaR based on:
        CVaR_beta(x) = min_zeta [ zeta + 1/(1-beta) * mean(max(x-zeta, 0)) ]

    Returns
    -------
    cvar_value : float
    zeta_star : float
    """
    if not (0 <= beta < 1):
        raise ValueError("beta must satisfy 0 <= beta < 1")

    x = np.asarray(losses, dtype=float).ravel()
    candidates = np.unique(x)

    # Evaluate the Rockafellar-Uryasev objective on all breakpoints
    tail = np.maximum(x[:, None] - candidates[None, :], 0.0)
    obj = candidates + tail.mean(axis=0) / (1.0 - beta)

    idx = int(np.argmin(obj))
    return float(obj[idx]), float(candidates[idx])


def calculate_cvar_term(data: pd.DataFrame, beta=0.9, r_df: pd.DataFrame = None):
    """
    Third aggregation term: CVaR-based risk term A_risk.

    Formula:
        A_j^risk = min_zeta [ zeta + 1/((1-beta)S) * sum_s (r_{s,j}-zeta)_+ ]

    Returns
    -------
    A_risk : pd.Series
        CVaR-based risk score for each algorithm
    zeta_star : pd.Series
        optimal zeta for each algorithm
    """
    if r_df is None:
        r_df = compute_normalized_regret(data)

    scores = {}
    zetas = {}

    for model in r_df.columns:
        cvar_value, zeta_star = _empirical_cvar(r_df[model].to_numpy(), beta=beta)
        scores[model] = cvar_value
        zetas[model] = zeta_star

    return (
        pd.Series(scores, name="A_risk"),
        pd.Series(zetas, name="zeta_star"),
    )


def calculate_cgre(data: pd.DataFrame,
                   capacities,
                   theta=(1/3, 1/3, 1/3),
                   beta=0.9):
    """
    Final composite score:
        G_j = theta1 * A_j^opt + theta2 * A_j^cap + theta3 * A_j^risk

    Parameters
    ----------
    data : pd.DataFrame
        rows = sites, columns = algorithms/models
    capacities : array-like or pd.Series
        installed capacities of sites
    theta : tuple/list of length 3
        weights for (A_opt, A_cap, A_risk)
    beta : float
        CVaR confidence level, usually 0.8 / 0.9 / 0.95

    Returns
    -------
    result : pd.DataFrame
        columns = [A_opt, A_cap, A_risk, G], sorted by G ascending
    details : dict
        intermediate outputs
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    theta = np.asarray(theta, dtype=float).ravel()
    if len(theta) != 3:
        raise ValueError("theta must contain 3 weights: (theta1, theta2, theta3)")
    if (theta < 0).any():
        raise ValueError("theta must be non-negative")
    if np.isclose(theta.sum(), 0):
        raise ValueError("theta sum cannot be zero")

    # Normalize theta automatically
    theta = theta / theta.sum()

    A_opt, w_opt, r_df = calculate_opt_term(data)
    A_cap, w_cap = calculate_capacity_term(data, capacities, r_df=r_df)
    A_risk, zeta_star = calculate_cvar_term(data, beta=beta, r_df=r_df)

    result = pd.concat([A_opt, A_cap, A_risk], axis=1)
    result["G"] = (
        theta[0] * result["A_opt"]
        + theta[1] * result["A_cap"]
        + theta[2] * result["A_risk"]
    )

    # smaller is better
    # result = result.sort_values("G", ascending=True)

    details = {
        "normalized_regret": r_df,
        "optimized_site_weights": w_opt,
        "capacity_weights": w_cap,
        "zeta_star": zeta_star,
        "theta": theta,
        "beta": beta,
    }

    return result, details

def get_metric_block(file_path: str, metric_name: str):

    raw_df = pd.read_excel(file_path, header=None)

    metric_row = raw_df.iloc[0, :]

    metric_positions = [i for i, value in enumerate(metric_row)
                        if str(value).strip() == metric_name]

    if not metric_positions:
        raise ValueError(f"Metric '{metric_name}' was not found in the first header row.")

    start_col = metric_positions[0]

    end_col = raw_df.shape[1] - 1
    for col in range(start_col + 1, raw_df.shape[1]):
        cell_value = raw_df.iat[0, col]
        if pd.notna(cell_value) and str(cell_value).strip() != "":
            end_col = col - 1
            break

    algorithm_names = raw_df.iloc[1, start_col:end_col + 1].tolist()

    station_names = raw_df.iloc[2:, 0].tolist()

    data_block = raw_df.iloc[2:, start_col:end_col + 1].copy()

    data_block.columns = algorithm_names
    data_block.index = station_names

    data_block = data_block.loc[data_block.index.notna()]

    data_block = data_block.apply(pd.to_numeric, errors='coerce')

    return data_block

def plot_performance_metrics_with_top_labels_centered(data, num_metrics, num_models,save_name):

    models = data.index.tolist()
    metrics = data.columns.tolist()  

    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'         
    plt.rcParams['axes.labelweight'] = 'bold'   
    plt.rcParams['axes.titleweight'] = 'bold'   
    plt.rcParams['figure.titleweight'] = 'bold'  

    assert len(metrics) == num_metrics, f"Expected {num_metrics} metrics, but got {len(metrics)}"
    assert len(models) == num_models, f"Expected {num_models} models, but got {len(models)}"

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist() 
    angles += angles[:1] 

    # viridis  Spectral   coolwarm  RdYlGn  RdYlBu  RdBu  RdGy  PuOr  BrBG PRGn  PiYG
    # color_map = cm.viridis(np.linspace(0, 1, num_metrics))  
    color_map = cm.RdYlGn(np.linspace(0, 1, num_metrics))  

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    inner_radius = 0.2 

    width = (2 * np.pi / num_metrics) / num_models 

    for i, metric in enumerate(metrics):
        values = data[metric].tolist()
        
        angle = angles[i]  
        # print(len(values))
        for j, value in enumerate(values):
            # print(j)

            angle_offset = angle + j * width

            ax.bar(angle_offset, value + inner_radius, bottom=inner_radius, width=width, label=models[j] if i == 0 else "",
                   alpha=0.6, color=color_map[i])

            x_label = angle_offset
            y_label = value + inner_radius + 0.3  
            if j<=len(values)-1:
                ax.text(x_label, y_label, models[j], 
                        ha='center', va='center', fontsize=10, rotation=angle)  
            # if j==len(values)-1:
            #     # print(j)
            #     ax.text(x_label, y_label, models[j], 
            #             ha='center', va='center', fontsize=10.5, rotation=angle,color='red') 

    ax.set_yticklabels([]) 
    ax.set_xticks([])  

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=10, label=metrics[i]) 
        for i in range(num_metrics)
    ]
    ax.legend(handles=legend_elements, title='Metrics', bbox_to_anchor=(0.95, 0.9), loc='best')
    ax.spines['polar'].set_visible(False)
    # plt.savsefig(save_name+".svg", bbox_inches='tight', dpi=600)
    plt.tight_layout()
    plt.show()