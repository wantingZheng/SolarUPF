import argparse
from typing import List
from utils.metrics1 import cacluate_interval_score,evaluate_regress
from utils.tools import converge
from utils.build_metric_summary_df import build_metric_summary_df
from data.data_process import DataLoader
from data.data_process_external import DataLoaderExternal
from models.custom_lgbm import CustomLightgbm
from models.custom_hgbt import CustomHGBoost
from models.custom_catb import CustomCatBoost
from models.custom_tabpfn import CustomTabPFN
from models.custom_ngbt import CustomNGBoost
from models.custom_baye import CustomBaye
from models.custom_tabm1 import CustomTABM1,TabMConfig
from models.custom_lstmr import CustomLSTM1
from models.custom_mlp import CustomMLP
from models.custom_tabilc import CustomTabICL
from models.upe3D import UPE3AModel
from models.upe2A import UPE2AModel

from pathlib import Path
import numpy as np
import pandas as pd
import gc
import os
from utils.tools import set_seed
import torch
import matplotlib.pyplot as plt

set_seed(3407)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
  
day_set=30    # train day set inclunding(train_valid_test)

zhixin=np.array([0,0.3,0.5,0.7,0.9]) # prediction confidence levelsnfidential 
# the first value must be 0, corresponding to the 0.5 quantile (deterministic point prediction),
# while the remaining confidence levels can be flexibly specified in ascending order,
#  e.g., [0, 0.3, 0.9] or [0, 0.1, 0.2, 0.3, 0.6, 0.9]

# zhixin=[0,0.3,0.5,0.7,0.9]
zhixin1=zhixin[1:]

data_set_list = ['data_JS', 'data_FBS', 'data_NG', 'data_AU', 'data_GED']
# data_set_list = ['data_JS']
# data_set_list = ['data_JS', 'data_NG', 'data_AU', 'data_GED']
# data_set_list = ['data_JS', 'data_NG', 'data_AU']
# Geographic information for different datasets
site_config = {
    'data_FBS': {
        'std_meridian_deg': 120,
        'longitude': 104.0,
        'latitude': 37.0
    },
    'data_JS': {
        'std_meridian_deg': 120,
        'longitude': 93.4,
        'latitude': 36.0
    },
    'data_NG': {
        'std_meridian_deg': 120,
        'longitude': 113.4,
        'latitude': 38.0
    },
    'data_GED': {
        'std_meridian_deg': 0,
        'longitude': 133.87,
        'latitude': -23.7
    },
    'data_AU': {
        'std_meridian_deg': 120,
        'longitude': 130.98,
        'latitude': -25.24
    }
}


# method_choose_list=['TabPFN','TABM','LightGBM','Baye','HGBoost','NGBoost','LSTM','KAN','MLP','UPF3']
method_choose_list=['TabPFN','TABM','NGBoost','TabICL','Baye','LightGBM','HGBoost','MLP','LSTM','KAN','UPF2A','UPF2B','UPF2C','UPF3A','UPF3B','UPF3C']

# method_choose_list=['TabPFN','TABM','NGBoost','TabICL','UPF3A']
method_choose_list1=[]

prdict_y_list_all=[]
true_y_list_all=[]
lower_y_list_all=[]
up_y_list_all=[]

output_df_ngboost_all=[]
point_predict_all=[]
interval_predict_all=[]
df_test_list_all=[]

prdict_y_list_external_all = []
true_y_list_external_all = []
lower_y_list_external_all = []
up_y_list_external_all = []
point_predict_external_all = []
interval_predict_external_all = []
df_test_list_external_all=[]
index=np.arange(0,len(method_choose_list))  #method choose
# index=[0,1,2,3,4,5,6]

# index=[0,1,2,3]
# index=[0,1,2,4,10,11,12,13,14]
# index=[17]
for index_set in index:
    method_choose=method_choose_list[index_set]
    method_choose_list1.append(method_choose)
    station_name_list=[] #record the station_name of all eara
    pv_station_data_list=[] #record the data of all station
    point_predict=[]  
    interval_predict=[]  
    true_y_list=[]
    prdict_y_list=[]
    lower_y_list=[]
    up_y_list=[]

    true_y_list_external = []
    prdict_y_list_external = []
    lower_y_list_external = []
    up_y_list_external = []
    point_predict_external = []
    interval_predict_external = []

    model_list=[]
    standard_list=[]
    mae_vaild=0
    station_idx1=0
    for data_test_name in data_set_list:
        project_root = Path.cwd()
        dataset_folder_path = project_root / 'datasets' / data_test_name

        print(dataset_folder_path)
        
        dataset_paths = os.listdir(dataset_folder_path)
        std_meridian_deg = site_config[data_test_name]['std_meridian_deg']
        longitude = site_config[data_test_name]['longitude']
        latitude = site_config[data_test_name]['latitude']


        for station_idx in range(0,len(dataset_paths)):
        # for station_idx in range(8,9):
            fafian_num=station_idx
            station_idx1=station_idx1+1; 

            dataset_path = dataset_paths[station_idx]
            # print(dataset_path) 

            station_name = dataset_path[:-5] 
            # print(station_name) 
            station_name_list.append(station_name)

            # df_all = pd.read_excel(dataset_folder_path + '/' + dataset_path)
            df_all = pd.read_excel(dataset_folder_path / dataset_path)

            df_process=df_all.dropna()

            df_process['row_index'] =np.arange(len(df_process))

            df_augmented = df_process.copy()

            method_choose1=method_choose
        
            # df_train_list=df_process[df_process['row_index']<=day_set*0.8*96]

            # df_vaild_list=df_process[(df_process['row_index']> day_set*0.8*96) & (df_process['row_index']<=0.9*day_set*96)]
            
            # df_test_list=df_process[(df_process['row_index']> day_set*0.9*96) & (df_process['row_index']<=day_set*96)]
            
            total_main = int(day_set * 96)
            train_end = int(day_set * 0.8 * 96)
            valid_end = int(day_set * 0.9 * 96)
            test_end  = int(day_set * 1.0 * 96)
            ext_end   = int(day_set * 1.1 * 96)

            df_train_list = df_process[(df_process['row_index'] >= 0) & (df_process['row_index'] < train_end)]
            df_vaild_list = df_process[(df_process['row_index'] >= train_end) & (df_process['row_index'] < valid_end)]
            df_test_list = df_process[(df_process['row_index'] >= valid_end) & (df_process['row_index'] < test_end)]
            df_test_list_external = df_process[(df_process['row_index'] >= test_end) & (df_process['row_index'] < ext_end)]
            
            drop_column=['row_index']

            df_train_list1=df_train_list.copy()
            df_train_list1=df_train_list1.drop(drop_column , axis=1)

            df_vaild_list1=df_vaild_list.copy()
            df_vaild_list1=df_vaild_list1.drop(drop_column , axis=1)

            df_test_list1=df_test_list.copy()
            df_test_list1=df_test_list1.drop(drop_column , axis=1)

            df_test_list_external1=df_test_list_external.copy()
            df_test_list_external1=df_test_list_external1.drop(drop_column , axis=1)
            
            dataloader = DataLoader(latitude,longitude,std_meridian_deg,df_train_list1,df_vaild_list1,df_test_list1)

            (train_x, train_y, train_x_nor,train_y_nor,
                    train_aug_x, train_aug_y,train_aug_x_nor,train_aug_y_nor,
                    val_x, val_y,val_x_nor, val_y_nor,
                    val_aug_x, val_aug_y,val_aug_x_nor, val_aug_y_nor,
                    test_x, test_y,test_x_nor, test_y_nor,
                    test_aug_x, test_aug_y, test_aug_x_nor, test_aug_y_nor,
                    y_mean, y_std) = dataloader.get_dataset()
            

            train_aug_x_mean = train_aug_x.mean(0)
            train_aug_x_std = train_aug_x.std(0)
            train_aug_x_std[train_aug_x_std == 0] = 1.0

            dataloader_external = DataLoaderExternal(
            latitude=latitude,
            longitude=longitude,
            std_meridian_deg=std_meridian_deg,
            external_df=df_test_list_external1,
            aug_x_mean=train_aug_x_mean,
            aug_x_std=train_aug_x_std,
            has_target=True)

            test_aug_x_ext_nor, test_y_external = dataloader_external.get_dataset1()
            # dataloader_external = DataLoader_external(latitude,longitude,std_meridian_deg,df_test_list_external1.iloc[:,:-1],y_mean_aug,y_std_aug)
            
            # test_aug_x_ext = dataloader_external.get_dataset1()
            # test_y_external=df_test_list_external1.iloc[:,-1]
            
            if method_choose1=='Baye':
                beya_model = CustomBaye(zhixin)
                beya_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)
                y_middle, upper, lower=beya_model.predict(test_aug_x_nor, y_mean, y_std)

                y_middle_external, upper_external, lower_external=beya_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='LightGBM':
                lgbm_model = CustomLightgbm(zhixin)
                lgbm_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)
                y_middle, upper, lower=lgbm_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=lgbm_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='TabPFN':
                tabpfn_au_model = CustomTabPFN(zhixin)
                tabpfn_au_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor )
                y_middle, upper, lower=tabpfn_au_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=tabpfn_au_model.predict(test_aug_x_ext_nor, y_mean, y_std)
            
            elif method_choose1=='TabICL':
                tabicl_model = CustomTabICL(zhixin)
                tabicl_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor )
                y_middle, upper, lower=tabicl_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=tabicl_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='CatBoost':
                catboost_model = CustomCatBoost(zhixin)
                catboost_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)
                # catboost_model.train( train_aug_x_nor, train_y_nor)
                y_middle, upper, lower=catboost_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=catboost_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='TABM':
                
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,                
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config)  
                tabm_model.train(train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)
                torch.cuda.empty_cache()
                y_middle, upper, lower=tabm_model.predict(test_aug_x_nor,y_mean,y_std)  
                y_middle_external, upper_external, lower_external=tabm_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='LSTM':
                gc.collect()
                torch.cuda.empty_cache()
                lstm_model = CustomLSTM1(zhixin,hidden_dim=64,num_layers=1,lr=0.001,batch_size=256,num_epochs=60,method_choose='LSTM')
                lstm_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)            
                y_middle, upper, lower=lstm_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=lstm_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='MLP':
                gc.collect()
                torch.cuda.empty_cache()
                mlp_model = CustomMLP(zhixin,input_dim=train_aug_x_nor.shape[1],hidden_dim=32,lr=0.001,batch_size=256,num_epochs=60,method_choose='MLP')
                mlp_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)                     
                y_middle, upper, lower=mlp_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=mlp_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='KAN':
                gc.collect()
                torch.cuda.empty_cache()
                kan_model = CustomMLP(zhixin,input_dim=train_aug_x_nor.shape[1],hidden_dim=32,lr=0.001,batch_size=256,num_epochs=60,method_choose='KAN')
                kan_model.train( train_aug_x_nor, train_y_nor,val_aug_x_nor, val_y_nor)                    
                y_middle, upper, lower=kan_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=kan_model.predict(test_aug_x_ext_nor, y_mean, y_std)

                    
            elif method_choose1=='NGBoost':

                ngboost_model = CustomNGBoost(zhixin)
                ngboost_model.train( train_aug_x_nor, train_y_nor, val_aug_x_nor,val_y_nor)
                y_middle, upper, lower=ngboost_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=ngboost_model.predict(test_aug_x_ext_nor, y_mean, y_std)
                
            elif method_choose1=='HGBoost':
                # lgbm = CustomLightgbm(zhixin)
                hgboost_model = CustomHGBoost(zhixin)
 
                hgboost_model.train( train_aug_x_nor, train_y_nor, val_aug_x_nor,val_y_nor)
                y_middle, upper, lower=hgboost_model.predict(test_aug_x_nor, y_mean, y_std)            
                y_middle_external, upper_external, lower_external=hgboost_model.predict(test_aug_x_ext_nor, y_mean, y_std)
 
            elif method_choose1=='UPF2A':
                gc.collect()
                torch.cuda.empty_cache()
                tabicl_model = CustomTabICL(zhixin)
                tabpfn_model = CustomTabPFN(zhixin)
                UPE2A_model=UPE2AModel(tabicl_model,tabpfn_model,zhixin)

                UPE2A_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)

                y_middle, upper, lower=UPE2A_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE2A_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='UPF2B':
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,               
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config)  
                ngb_model = CustomNGBoost(zhixin)

                UPE2B_model=UPE2AModel(ngb_model,tabm_model,zhixin)

                UPE2B_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)
                y_middle, upper, lower=UPE2B_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE2B_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='UPF2C':
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,                
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config)  
                # ngb_model = CustomNGBoost(zhixin)
                tabpfn_model = CustomTabPFN(zhixin)
                UPE2C_model=UPE2AModel(tabpfn_model,tabm_model,zhixin)
                UPE2C_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)
                y_middle, upper, lower=UPE2C_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE2C_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='UPF3A':
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,              
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config)  
                ngb_model = CustomNGBoost(zhixin)
                tabpfn_model = CustomTabPFN(zhixin)
                tabpfn_aug_model = CustomTabPFN(zhixin)

                UPE3A_model=UPE3AModel(ngb_model,tabpfn_model,tabm_model,zhixin)

                UPE3A_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)

                y_middle, upper, lower=UPE3A_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE3A_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='UPF3B':
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,                
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config) 
                mlp_model = CustomMLP(zhixin,input_dim=train_aug_x_nor.shape[1],hidden_dim=32,lr=0.001,batch_size=256,num_epochs=60,method_choose='MLP')                 
                # ngb_model = CustomNGBoost(zhixin)
                tabpfn_model = CustomTabPFN(zhixin)
                tabpfn_aug_model = CustomTabPFN(zhixin)

                UPE3A_model=UPE3AModel(mlp_model,tabpfn_model,tabm_model,zhixin)

                UPE3A_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)

                y_middle, upper, lower=UPE3A_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE3A_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            elif method_choose1=='UPF3C':
                gc.collect()
                torch.cuda.empty_cache()
                
                config = TabMConfig(
                task_type='regression',
                n_num_features=train_aug_x_nor.shape[1],
                cat_cardinalities=None,
                n_classes=None,
                num_embedding_type='none',   # 'none'|'linear_relu'|'periodic'|'piecewise'
                # regression_label_stats=regression_label_stats,
                lr=5e-3, weight_decay=3e-4,
                batch_size=256, num_epochs=50,
                gradient_clipping_norm=1.0,
                enable_amp=False,               
                compile_model=False)
                tabm_model = CustomTABM1(zhixin,config) 
                # ngb_model = CustomNGBoost(zhixin)
                tabpfn_model = CustomTabPFN(zhixin)
                tabpicl_model = CustomTabICL(zhixin)

                UPE3A_model=UPE3AModel(tabpicl_model,tabpfn_model,tabm_model,zhixin)

                UPE3A_model.train(train_aug_x_nor,train_y_nor,val_aug_x_nor,val_y_nor, val_y, y_mean, y_std)

                y_middle, upper, lower=UPE3A_model.predict(test_aug_x_nor, y_mean, y_std)
                y_middle_external, upper_external, lower_external=UPE3A_model.predict(test_aug_x_ext_nor, y_mean, y_std)

            true_y_list.append(test_y)
            prdict_y_list.append(y_middle)
            lower_y_list.append(lower)
            up_y_list.append(upper)

            true_y_list_external.append(test_y_external)
            prdict_y_list_external.append(y_middle_external)
            lower_y_list_external.append(lower_external)
            up_y_list_external.append(upper_external)

            MAE,NRMSE,R2,MAPE,RMSE=evaluate_regress(y_middle, test_y)
            point_predict.append([MAE,NRMSE,R2,MAPE,RMSE])

            MAE_ext,NRMSE_ext,R2_ext,MAPE_ext,RMSE_ext=evaluate_regress(y_middle_external, test_y_external)
            point_predict_external.append([MAE_ext,NRMSE_ext,R2_ext,MAPE_ext,RMSE_ext])

            quantile =1-(1-zhixin1)/2  
            quantile1 =(1-zhixin1)/2  
            cacluate_interval=np.ones((5,len(quantile)))        
            for i in range(len(quantile)):
                        # confidence_level = 0.95
                        confidence_level=zhixin1[i]
                        alpha=1-confidence_level
                        y_true=test_y
                        lower_bounds =lower[:,i]
                        upper_bounds=upper[:,i]
                        picp_result,pinaw_result,cwc_result,ql_result,interval_score_result=cacluate_interval_score(y_true, lower_bounds, upper_bounds, 10, alpha)

                        cacluate_interval[:,i]=[picp_result,pinaw_result,cwc_result,ql_result,interval_score_result]

            interval_predict.append(cacluate_interval)
            df_test_list_all.append(df_test_list)

            cacluate_interval_external=np.ones((5,len(quantile)))        
            for i in range(len(quantile)):
                        # confidence_level = 0.95
                        confidence_level=zhixin1[i]
                        alpha=1-confidence_level
                        y_true=test_y_external
                        lower_bounds =lower_external[:,i]
                        upper_bounds=upper_external[:,i]
                        picp_result,pinaw_result,cwc_result,ql_result,interval_score_result=cacluate_interval_score(y_true, lower_bounds, upper_bounds, 10, alpha)

                        cacluate_interval_external[:,i]=[picp_result,pinaw_result,cwc_result,ql_result,interval_score_result]

            interval_predict_external.append(cacluate_interval_external)
            # df_test_list_external_all.append(df_test_list_external)

            output_df=converge(station_name_list,df_test_list,true_y_list,prdict_y_list,up_y_list,lower_y_list,zhixin1) #概率区间预测汇集结果        
            name_str=method_choose+'_predict.xlsx'
            # output_df.to_excel(name_str)
    
    true_y_list_all.append(true_y_list)  
    prdict_y_list_all.append(prdict_y_list)      
    lower_y_list_all.append(lower_y_list)
    up_y_list_all.append(up_y_list)    
    output_df_ngboost_all.append(output_df)
    point_predict_all.append(point_predict)
    interval_predict_all.append(interval_predict)

    true_y_list_external_all.append(true_y_list_external)  
    prdict_y_list_external_all.append(prdict_y_list_external)      
    lower_y_list_external_all.append(lower_y_list_external)
    up_y_list_external_all.append(up_y_list_external)    
    point_predict_external_all.append(point_predict_external)
    interval_predict_external_all.append(interval_predict_external)

    NRMSE_array=np.zeros(len(point_predict))
    for i in range(len(point_predict)):
        NRMSE_array[i]=point_predict[i][1]
    print('*******************************************')
    print(method_choose+'_NRMSE')
    print(NRMSE_array)
 
    interval_MCE_array=np.zeros(len(point_predict))
    for i in range(len(interval_predict)):
        for j in range(0,4):
            interval_MCE_array[i]=interval_MCE_array[i]+interval_predict[i][3,j]
    interval_MCE_array1=interval_MCE_array/4

    print('*******************************************')
    print(method_choose+'_IS')
    print(interval_MCE_array1)

# Metric labels

metrics_label_list = [
    'MAE', 'NRMSE', 'R2', 'MAPE', 'RMSE',
    'PICP', 'PINAW', 'CWC', 'IS', 'QL'
]

result2_df_last = build_metric_summary_df(
    metrics_label_list=metrics_label_list,
    point_predict_all=point_predict_all,
    interval_predict_all=interval_predict_all,
    station_name_list=station_name_list,
    method_choose_list1=method_choose_list1
)

print(result2_df_last)
data_test_name1 ='testday_' + str(day_set) + '_interval_result1.xlsx'
result2_df_last.to_excel(data_test_name1)


result2_df_last_external = build_metric_summary_df(
    metrics_label_list=metrics_label_list,
    point_predict_all=point_predict_external_all,
    interval_predict_all=interval_predict_external_all,
    station_name_list=station_name_list,
    method_choose_list1=method_choose_list1
)

# print(result2_df_last)
data_test_name2 ='external_testday_' + str(day_set) + '_interval_result1.xlsx'
result2_df_last_external.to_excel(data_test_name2)
