import numpy as np
from sklearn.metrics import r2_score

def picp(y_true, lower_bounds, upper_bounds):
    # PICP 
    in_interval = np.logical_and(y_true >= lower_bounds, y_true <= upper_bounds)
    
    return np.mean(in_interval)

def interval_score(y_true, lower_bounds, upper_bounds, alpha):
    # Interval Score (IS)
    n = len(y_true)
    score = (upper_bounds - lower_bounds) + (2 / alpha) * ((lower_bounds - y_true) * (y_true < lower_bounds) +
                                                           (y_true - upper_bounds) * (y_true > upper_bounds))
    return np.mean(score)

def pinaw(lower_bounds, upper_bounds, y_true):
    # PINAW 
    y_range = np.max(y_true) - np.min(y_true)
    width = upper_bounds - lower_bounds
    
    return np.mean(width) / y_range

# def cwc(y_true, lower_bounds, upper_bounds, gamma, alpha):
#     picp_val = picp(y_true, lower_bounds, upper_bounds)
#     pinaw_val = pinaw(lower_bounds, upper_bounds, y_true)
#     penalty = gamma * max(0, (1-alpha- picp_val))
    
#     return pinaw_val * (1 + penalty)
def cwc(y_true, lower_bounds, upper_bounds, gamma=10, alpha=0.9):
    """
    CWC with Heaviside step penalty (Khosravi-style)

    Parameters
    ----------
    mu     : target coverage probability (e.g., 0.9)
    eta    : penalty strength (default 50)
    gamma  : penalty steepness (default 10)
    """
    picp_val  = picp(y_true, lower_bounds, upper_bounds)
    pinaw_val = pinaw(lower_bounds, upper_bounds, y_true)

    mu=1-alpha

    # Heaviside step function
    H = 1.0 if (mu - picp_val) > 0 else 0.0

    # Exponential penalty term
    eta=0.5
    penalty = eta * H * np.exp(-gamma * (picp_val - mu))

    return pinaw_val * (1.0 + penalty)

def quantile_loss_Q(y_true, lower_bounds, upper_bounds, q):
    # Quantile Loss 
    lower_loss = q * (y_true - lower_bounds) * (y_true < lower_bounds)
    upper_loss = (1 - q) * (upper_bounds - y_true) * (y_true > upper_bounds)    
    return np.mean(lower_loss + upper_loss)

def interval_violation_loss(y_true, lower_bounds, upper_bounds, q):
    """
    Interval violation loss (non-negative)

    y_true, lower_bounds, upper_bounds: numpy arrays
    q: quantile weight (e.g., 0.9)
    """
    lower_violation = np.maximum(lower_bounds - y_true, 0.0)
    upper_violation = np.maximum(y_true - upper_bounds, 0.0)

    loss = q * lower_violation + (1 - q) * upper_violation
    return np.mean(loss)

def mean_interval_score(y_true, lower_bounds, upper_bounds, alpha):
    mis = (upper_bounds - lower_bounds) + (2 / alpha) * np.maximum(0, lower_bounds - y_true) + (2 / alpha) * np.maximum(0, y_true - upper_bounds)
    
    return np.mean(mis)

###################################################指定评价函数

def evaluate_regress(y_pre, y_true):
   
    MAE=np.sum(np.abs(y_pre-y_true))/len(y_true)

    
    # MCE=np.sum(np.abs((y_pre-y_true)/max(y_true)))/len(y_true)

    # NRMSE
    rmse = np.sqrt(np.mean((y_pre - y_true)**2))

    NRMSE = rmse / (np.max(y_true) - np.min(y_true) + 1e-12)

    MAPE=np.sum(np.abs(y_pre-y_true)/(y_true+1e-5))/len(y_true)

    MSE=np.sum((y_pre-y_true) ** 2)/len(y_true)
    
    RMSE=np.sqrt(MSE)

    R2=r2_score(y_true, y_pre)

    return MAE,NRMSE,R2,MAPE,RMSE

def cacluate_interval_score(y_true, lower_bounds, upper_bounds,gamma, alpha):
    #y_true [N*T] lower_bounds[N*T*q]
    '''
    Prediction Interval Coverage Probability (PICP): The closer the PICP is to the target coverage level (e.g., 95%), the better. PICP represents the proportion of true observations that fall within the predicted interval. A value closer to the nominal confidence level indicates that the prediction interval more accurately captures the true values.

    Interval Score (IS): Smaller values indicate better performance. This metric jointly considers the width and the accuracy of the prediction interval. A lower IS implies a narrower interval with higher reliability.

    Prediction Interval Normalized Average Width (PINAW): Smaller values are preferred. PINAW measures the average width of the prediction interval after normalization, where a smaller value indicates a tighter interval.

    Coverage Width-Based Criterion (CWC): Smaller values indicate better performance. CWC integrates both coverage reliability and interval sharpness, where a lower value reflects a narrower interval while satisfying the desired coverage requirement.

    Overall, PICP should be as close as possible to the target coverage level, while the other metrics are generally minimized to achieve narrow and reliable prediction intervals that adequately capture most of the true observations.

    Here, gamma denotes the penalty coefficient in the CWC formulation.
    '''
    
    picp_result = picp(y_true, lower_bounds, upper_bounds)
    interval_score_result = interval_score(y_true, lower_bounds, upper_bounds,alpha)
    pinaw_result = pinaw(lower_bounds, upper_bounds, y_true)
    gamma=10
    cwc_result = cwc(y_true, lower_bounds, upper_bounds,gamma, alpha)
    ql_result = interval_violation_loss(y_true, lower_bounds, upper_bounds,alpha)
    # mis_result = mean_interval_score(y_true, lower_bounds, upper_bounds,alpha)
    # print("######################################################################################################")
    # print("PICP:", picp_result)
    # print("Interval Score:", interval_score_result)
    # print("PINAW:", pinaw_result)
    # print("CWC:", cwc_result)
    # print("Quantile Loss:", ql_result)
    # print("Mean Interval Score:", mis_result)

    return picp_result,pinaw_result,cwc_result,interval_score_result,ql_result

