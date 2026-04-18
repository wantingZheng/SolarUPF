import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Dict, Tuple

from utils.tools import augment_dataset_base


class DataLoader:
    def __init__(self,latitude,longitude,std_meridian_deg,train_df,vaild_df,test_df=None) -> None:
        # self.data_path = data_path
        # self.aug = aug
        self.train_df=train_df
        self.vaild_df=vaild_df
        self.test_df=test_df
        self.latitude=latitude
        self.longitude=longitude
        self.std_meridian_deg=std_meridian_deg
    
    def calculate_mean_std(self, x: np.ndarray, y: np.ndarray):
        x_mean, y_mean = x.mean(0), y.mean(0)
        x_std, y_std = x.std(0), y.std(0)
        
        return [x_mean, x_std, y_mean, y_std]
    
    def normalize(self, x: np.ndarray, y: np.ndarray, norm_info: List) -> Tuple[np.ndarray, np.ndarray]:
        x_mean, x_std, y_mean, y_std = norm_info
        norm = lambda x, mean, std: (x - mean) / std
        norm_x, norm_y = norm(x, x_mean, x_std), norm(y, y_mean, y_std)
        
        return norm_x, norm_y
        
    def create_dataset(self, data_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        drop_columns=data_df.columns[0]
        data_df=data_df.drop(drop_columns , axis=1)
        x = data_df.values[:,:-1]
        y = data_df.values[:,-1]
        
        return x, y
    
    def augment_dataset(self, data_df, latitude,longitude,std_meridian_deg) -> pd.DataFrame:
        # std_meridian_deg : Standard meridian of the local time zone, used to convert local clock time into solar time.
        #            For example, China typically uses 120°E (UTC+8), while UTC is 0°.

        data_df.rename(columns={data_df.columns[0]:'timestamp'}, inplace=True)
        data_df.rename(columns={data_df.columns[-1]:'power'}, inplace=True) 
        data_df['TIMESTAMP'] = data_df['timestamp']        
        data_df['TIMESTAMP'] = pd.to_datetime(data_df[data_df.columns[0]]) 
        # data_df['day_of_year'] = data_df['TIMESTAMP'].apply(lambda timestamp: timestamp.day_of_year)
        # data_df['declination_in_degree'] = data_df['day_of_year'].apply(calculate_declination_in_degree)  # delta, declination, 赤纬，太阳直射角度。
        # data_df['time_degree'] = data_df['TIMESTAMP'].apply(lambda timestamp: (timestamp.hour + timestamp.minute / 60 - 12) * 15)  # 时角，时间角度，12点为0°，每增加一小时，增加15°；减少一小时，减少15°。
        # data_df['sun_height'] = calculate_sun_height(latitude, data_df['declination_in_degree'].values, data_df['time_degree'].values)
        # data_df['power1'] = data_df['power']
        # data_df=data_df.drop(['power'] , axis=1)
        augment_dataset_base
        data_df_augment=augment_dataset_base(data_df,
        latitude,
        longitude,
        std_meridian_deg,
        I_sc = 1367.0,
        declination_mode= "spencer")

        data_df.drop(columns=['timestamp'], inplace=True)
        data_df.insert(0, 'TIMESTAMP', data_df.pop('TIMESTAMP'))

        data_df_merge = pd.concat([data_df, data_df_augment], axis=1)

        # 把 power 放到最后
        cols = [col for col in data_df_merge.columns if col != 'power'] + ['power']
        data_df_merge = data_df_merge[cols]
        
        return data_df_merge
    
    def prepare_dataset(self, data_df: pd.DataFrame, norm_info: List, norm_info_aug: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y = self.create_dataset(deepcopy(data_df))
        x_nor, y_nor = self.normalize(x, y, norm_info)

        aug_data_df = self.augment_dataset(data_df.copy(),self.latitude,self.longitude,self.std_meridian_deg)
        
        # aug_data_df = self.augment_dataset(data_df.copy(),self.latitude)
        aug_x, aug_y = self.create_dataset(aug_data_df)
        aug_x_nor, aug_y_nor = self.normalize(aug_x, aug_x, norm_info_aug)
        
        return x, y,x_nor, y_nor, aug_x, aug_y,aug_x_nor, aug_y_nor
    
    def get_dataset(self) -> Tuple:

        train_x, train_y =self.create_dataset(deepcopy(self.train_df))

        norm_info = self.calculate_mean_std(train_x, train_y)

        train_x_nor, train_y_nor = self.normalize(train_x, train_y, norm_info)
            
        aug_data_df = self.augment_dataset(self.train_df.copy(),self.latitude,self.longitude,self.std_meridian_deg)


        train_aug_x, train_aug_y =self.create_dataset(deepcopy(aug_data_df)) 

        norm_info_aug = deepcopy(norm_info)
        
        norm_info_aug[:2] = [train_aug_x.mean(0), train_aug_x.std(0)] # y不变，只是命名对齐

        train_aug_x_nor, train_aug_y_nor = self.normalize(train_aug_x, train_aug_x, norm_info_aug)   

        y_mean, y_std = norm_info[2:]

        # y_mean_aug, y_std_aug = norm_info_aug[2:]

        val_x, val_y,val_x_nor, val_y_nor, val_aug_x, val_aug_y,val_aug_x_nor, val_aug_y_nor = self.prepare_dataset(self.vaild_df, norm_info, norm_info_aug)

        if len(self.test_df)==0:
            test_x=0
            test_y=0
            test_x_nor=0
            test_y_nor=0
            test_aug_x=0
            test_aug_y=0
            test_aug_x_nor=0
            test_aug_y_nor=0
        else:
            test_x, test_y,test_x_nor, test_y_nor, test_aug_x, test_aug_y, test_aug_x_nor, test_aug_y_nor = self.prepare_dataset(self.test_df, norm_info, norm_info_aug)
              
        return (train_x, train_y, train_x_nor,train_y_nor,
                train_aug_x, train_aug_y,train_aug_x_nor,train_aug_y_nor,
                val_x, val_y,val_x_nor, val_y_nor,
                val_aug_x, val_aug_y,val_aug_x_nor, val_aug_y_nor,
                test_x, test_y,test_x_nor, test_y_nor,
                test_aug_x, test_aug_y, test_aug_x_nor, test_aug_y_nor,
                y_mean, y_std)