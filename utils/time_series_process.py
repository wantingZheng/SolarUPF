import numpy as np
import pandas as pd
def build_time_series_dataset(
    df,
    past_lags_num=12,          # past steps
    future_horizons_num=4,    # predict steps
    known_future_cols=None,       # 
    unknown_future_cols=None,     # 
    target_cols=None              # 
):
    future_horizons=np.arange(0,future_horizons_num)
    # past_target_lags=np.arange(0,past_target_lags_num)
    past_lags=np.arange(1,past_lags_num+1)
    past_lags1=np.arange(1,past_lags_num-future_horizons_num+1)

    df_new = pd.DataFrame(index=df.index)

    # ---- ① unForecastable features  ----
    if unknown_future_cols is not None:
        for col in unknown_future_cols:
            for lag in past_lags:
                new_name = f"{col}-t-{lag}"
                df_new[new_name] = df[col].shift(lag)

    # ---- ② Forecastable features ----
    if known_future_cols is not None:
        for col in known_future_cols:
            for horizon in future_horizons:
                new_name = f"{col}-t+{horizon}"
                df_new[new_name] = df[col].shift(-horizon)
                
            #  t-k
            for lag in past_lags1:
                new_name = f"{col}-t-{lag}"
                df_new[new_name] = df[col].shift(lag)

    # ---- ③  Y ----
    df_y = pd.DataFrame(index=df.index)
    col1=target_cols
    for horizon in future_horizons:
        new_name = f"{col1}-t+{horizon}"
        df_y[new_name] = df[col1].shift(-horizon)

    # ---- ④  ----
    df_all = pd.concat([df_new, df_y], axis=1)

    # ---- ⑤ delete NaN----
    df_all = df_all.dropna().reset_index(drop=False)

    return df_all,df_new, df_y
