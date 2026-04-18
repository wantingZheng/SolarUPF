import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   

def calculate_declination_in_degree(day_of_year, mode="spencer"):
    day = np.asarray(day_of_year, dtype=float)

    if mode == "paper":
        return 23.45 * np.sin(2 * np.pi * (284.0 + day) / 365.0)

    elif mode == "spencer":
        b = 2 * np.pi * (day - 1.0) / 365.2422
        delta = (
            0.006918
            - 0.399912 * np.cos(b)
            + 0.070257 * np.sin(b)
            - 0.006758 * np.cos(2 * b)
            + 0.000907 * np.sin(2 * b)
            - 0.002697 * np.cos(3 * b)
            + 0.00148 * np.sin(3 * b)
        )  # rad
        return np.rad2deg(delta)

    else:
        raise ValueError("mode must be 'paper' or 'spencer'")


def calculate_equation_of_time_min(day_of_year):
    """
        B_t = 2*pi*(d-81)/364
        E_t = 9.87*sin(2B_t) - 7.53*cos(B_t) - 1.5*sin(B_t)
        B_t, E_t (min)
    """
    day = np.asarray(day_of_year, dtype=float)
    B_t = 2 * np.pi * (day - 81.0) / 364.0
    E_t = 9.87 * np.sin(2 * B_t) - 7.53 * np.cos(B_t) - 1.5 * np.sin(B_t)
    return B_t, E_t


def calculate_solar_time_hour(timestamps, longitude_deg, std_meridian_deg=120.0):
    """
    t_sol = t_loc + [4*(lambda - lambda_std) + E_t]/60

    - longitude_deg
    - std_meridian_deg
    """
    ts = pd.Series(pd.to_datetime(timestamps))
    clock_hour = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    day_of_year = ts.dt.dayofyear.to_numpy()

    B_t, E_t = calculate_equation_of_time_min(day_of_year)

    solar_time = clock_hour.to_numpy(dtype=float) + (
        4.0 * (longitude_deg - std_meridian_deg) + E_t
    ) / 60.0

    return solar_time, B_t, E_t


def calculate_solar_hour_angle_deg(solar_time_hour):
    """
    :
        omega = 15 * (t_sol - 12)
    """
    solar_time_hour = np.asarray(solar_time_hour, dtype=float)
    return 15.0 * (solar_time_hour - 12.0)


def calculate_sin_sun_height(latitude_deg, declination_deg, hour_angle_deg):
    """
     sin(h)
    sin(h) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(omega)
    """
    phi = np.deg2rad(float(latitude_deg))
    delta = np.deg2rad(np.asarray(declination_deg, dtype=float))
    omega = np.deg2rad(np.asarray(hour_angle_deg, dtype=float))

    sin_h = (
        np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.cos(omega)
    )

    return np.clip(sin_h, -1.0, 1.0)


def calculate_sun_height_deg(latitude_deg, declination_deg, hour_angle_deg):
    """
     h（degree）
    """
    sin_h = calculate_sin_sun_height(latitude_deg, declination_deg, hour_angle_deg)
    return np.rad2deg(np.arcsin(sin_h))


def calculate_I0_horizontal(day_of_year, sin_sun_height, I_sc=1367.0):
    """
    
        I0 = I_sc * (1 + 0.033*cos(2*pi*d/365)) * max(sin(h), 0)
    """
    day = np.asarray(day_of_year, dtype=float)
    sin_h = np.asarray(sin_sun_height, dtype=float)

    return (
        I_sc
        * (1.0 + 0.033 * np.cos(2.0 * np.pi * day / 365.0))
        * np.maximum(sin_h, 0.0)
    )


def calculate_k_sc(ghi, I0_horizontal, min_i0=20.0, epsilon=1e-6):
    ghi = np.asarray(ghi, dtype=float)
    I0 = np.asarray(I0_horizontal, dtype=float)

    k = np.full_like(I0, np.nan, dtype=float)
    mask = I0 > min_i0
    k[mask] = ghi[mask] / (I0[mask] + epsilon)
    return k


def augment_dataset_base(
    data_df: pd.DataFrame,
    latitude: float,
    longitude: float,
    std_meridian_deg: float = 120.0,
    I_sc: float = 1367.0,
    declination_mode: str = "spencer",
) -> pd.DataFrame:
    """
    Inputs:
        data_df          : Historical PV plant dataset
        latitude         : Latitude in degrees
        longitude        : Longitude in degrees (positive for east, negative for west)
        ghi_col          : Name of the GHI column; if not provided, k_sc will be returned as NaN
        std_meridian_deg : Local standard meridian in degrees; for UTC+8, use 120.0
        I_sc             : Solar constant, default is 1367
        declination_mode : Method for declination calculation, either 'spencer' or 'paper'

    Output:
        Returns the augmented DataFrame with engineered solar-geometry features
    """
    df = data_df.copy()

    # if timestamp_col is None:
    #     timestamp_col = df.columns[0]

    # if power_col is None and len(df.columns) > 1:
    #     power_col = df.columns[-1]

    # if power_col is not None and power_col in df.columns and power_col != "power":
    #     df = df.rename(columns={power_col: "power"})

    # time
    # df["TIMESTAMP"] = pd.to_datetime(df[timestamp_col])
    df["day_of_year"] = df["TIMESTAMP"].dt.dayofyear.astype(int)

    solar_time_hour, B_t, E_t = calculate_solar_time_hour(
        df["TIMESTAMP"],
        longitude_deg=longitude,
        std_meridian_deg=std_meridian_deg,
    )
    df["B_t"] = B_t
    df["E_t_min"] = E_t
    df["solar_time_hour"] = solar_time_hour

    # solar declination angle
    df["declination_deg"] = calculate_declination_in_degree(
        df["day_of_year"].to_numpy(),
        mode=declination_mode
    )

    # solar hour angle
    df["solar_hour_angle_deg"] = calculate_solar_hour_angle_deg(
        df["solar_time_hour"].to_numpy()
    )

    # solar elevation angle
    df["sin_sun_height"] = calculate_sin_sun_height(
        latitude_deg=latitude,
        declination_deg=df["declination_deg"].to_numpy(),
        hour_angle_deg=df["solar_hour_angle_deg"].to_numpy(),
    )
    df["sun_height_deg"] = np.rad2deg(
        np.arcsin(np.clip(df["sin_sun_height"].to_numpy(), -1.0, 1.0))
    )

    df["I0_horizontal"] = calculate_I0_horizontal(
        df["day_of_year"].to_numpy(),
        df["sin_sun_height"].to_numpy(),
        I_sc=I_sc,
    )

    df["I_sc"] = I_sc

    df["f_declination_deg"] = df["declination_deg"]
    df["f_solar_hour_angle_deg"] = df["solar_hour_angle_deg"]
    df["f_sin_sun_height"] = df["sin_sun_height"]
    df["f_I0_horizontal"] = df["I0_horizontal"]

    return df[[
            "f_declination_deg",
            "f_solar_hour_angle_deg",
            "f_sin_sun_height",
            "f_I0_horizontal",]]

# def calculate_sun_height(phi, delta, t):
#     sinh = np.sin(phi*np.pi / 180) * np.sin(delta*np.pi / 180) + np.cos(phi*np.pi / 180) * np.cos(delta*np.pi / 180) * np.cos(t*np.pi / 180)
#     return sinh

# def calculate_declination_in_degree(day_of_year):
#     # input: day of year, from 1 to 365/366. count from January 1th.
#     # output: declination in degree
#     b = 2*np.pi*(day_of_year-1)/365.2422  # radian
#     delta=0.006918-0.399912*np.cos(b)+0.070257*np.sin(b)-0.006758*np.cos(2*b)+0.000907*np.sin(2*b)-0.002697*np.cos(3*b)+0.00148*np.sin(3*b)  # in radian
#     delta_in_degree = delta * 180/np.pi
#     return delta_in_degree

def converge(station_name_list,df_test_list,true_test_y_list,prdict_y_list,up_y_list,lower_y_list,zhixin):
    repeated_codes = np.repeat(station_name_list, len(true_test_y_list[0]))

    time_index_data=df_test_list.copy()
    # time_index_data=df_test_list.iloc[int(len(df_test_list)/2):]
    time_index_df=pd.DataFrame(time_index_data[time_index_data.columns[0]])

    df_repeated_data = pd.concat([time_index_df] * len(true_test_y_list), ignore_index=True)

    # df1['TIMESTAMP']=df_repeated_data['TIMESTAMP']
    df_repeated_data.index=repeated_codes

    combined_matrix_lower = np.vstack(lower_y_list)
    combined_matrix_upper=np.vstack(up_y_list)

    combined_matrix_true=np.vstack(true_test_y_list)

    combined_matrix_predict=np.vstack(prdict_y_list)

    combined_matrix_true1 = combined_matrix_true.flatten()  
    combined_matrix_true2 = combined_matrix_true1.reshape(-1, 1)  


    combined_matrix_predict1 = combined_matrix_predict.flatten() 
    combined_matrix_predict2 = combined_matrix_predict1.reshape(-1, 1) 

    list_name1=[]
    list_name2=[]
    for i in range(0,len(zhixin)):
        quantile1 =(1-zhixin)/2 
        quantile =1-(1-zhixin)/2
        list_name1.append(str(np.round(quantile[i],2))+'_quantile_pre')
        list_name2.append(str(np.round(quantile1[i],2))+'_quantile_pre')

    df_list = [
        pd.DataFrame(combined_matrix_true2, columns=["true"]),
        pd.DataFrame(combined_matrix_predict2, columns=["predict"]),
        pd.DataFrame(combined_matrix_upper, columns=list_name1),
        pd.DataFrame(combined_matrix_lower, columns=list_name2)
    ]

    combined_df = pd.concat(df_list, axis=1)
    combined_df1=combined_df.reset_index(drop=True)
    df_repeated_data1=df_repeated_data.reset_index(drop=True)

    output_df1=pd.concat([df_repeated_data1,combined_df1],axis=1)

    output_df1.index=repeated_codes

    return output_df1