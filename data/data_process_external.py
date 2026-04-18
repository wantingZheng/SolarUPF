# data/data_process_external.py

import numpy as np
import pandas as pd
from utils.tools import augment_dataset_base

class DataLoaderExternal:
    def __init__(
        self,
        latitude,
        longitude,
        std_meridian_deg,
        external_df: pd.DataFrame,
        aug_x_mean: np.ndarray,
        aug_x_std: np.ndarray,
        has_target: bool = True
    ) -> None:
        """
        Data loader for external test data.

        Parameters
        ----------
        latitude : float
            Latitude of the PV site.
        longitude : float
            Longitude of the PV site.
        std_meridian_deg : float
            Standard meridian of the local time zone, used for solar-time-based feature augmentation.
        external_df : pd.DataFrame
            External test dataframe.
            If has_target=True, the expected format is:
                [timestamp, feature_1, ..., feature_n, power]
            If has_target=False, the expected format is:
                [timestamp, feature_1, ..., feature_n]
        aug_x_mean : np.ndarray
            Mean of the augmented training features, used to normalize external augmented features.
        aug_x_std : np.ndarray
            Standard deviation of the augmented training features, used to normalize external augmented features.
        has_target : bool, default=True
            Whether the external dataframe contains the target column.
        """
        self.external_df = external_df.copy()
        self.latitude = latitude
        self.longitude = longitude
        self.std_meridian_deg = std_meridian_deg
        self.aug_x_mean = np.asarray(aug_x_mean, dtype=float)
        self.aug_x_std = np.asarray(aug_x_std, dtype=float)
        self.has_target = has_target

        # Avoid division by zero when a feature has zero standard deviation
        self.aug_x_std[self.aug_x_std == 0] = 1.0

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize external augmented features using the training-feature statistics.
        """
        return (x - self.aug_x_mean) / self.aug_x_std

    def augment_dataset(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the same feature augmentation as the training DataLoader.

        The first column is treated as the timestamp column.
        If has_target=True, the last column is treated as the target column 'power'.
        """        
        
        data_df = data_df.copy()

        # Rename the first column to 'timestamp' for consistency
        data_df.rename(columns={data_df.columns[0]: 'timestamp'}, inplace=True)


        # If the external set contains the target, rename the last column to 'power'
        if self.has_target:
            data_df.rename(columns={data_df.columns[-1]: 'power'}, inplace=True)
        # Safety check
        # Build a unified datetime column used by the augmentation function
        data_df['TIMESTAMP'] = pd.to_datetime(data_df['timestamp'])

        if 'TIMESTAMP' not in data_df.columns:
            raise KeyError("Column 'TIMESTAMP' was not created successfully.")
        # Generate augmented solar/physics-related features
        data_df_augment = augment_dataset_base(
            data_df,
            self.latitude,
            self.longitude,
            self.std_meridian_deg,
            I_sc=1367.0,
            declination_mode="spencer"
        )

        # Remove the original temporary timestamp column
        data_df.drop(columns=['timestamp'], inplace=True)

        # Move TIMESTAMP to the first column
        data_df.insert(0, 'TIMESTAMP', data_df.pop('TIMESTAMP'))

        # Concatenate original features and augmented features
        data_df_merge = pd.concat(
            [data_df.reset_index(drop=True), data_df_augment.reset_index(drop=True)],
            axis=1
        )

        # If target exists, move 'power' to the last column
        if self.has_target and ('power' in data_df_merge.columns):
            cols = [col for col in data_df_merge.columns if col != 'power'] + ['power']
            data_df_merge = data_df_merge[cols]

        return data_df_merge

    def get_dataset1(self):
        """
        Build the external augmented dataset.

        Returns
        -------
        If has_target=True:
            aug_x_nor : np.ndarray
                Normalized augmented features for external testing.
            aug_y : np.ndarray
                True target values of the external test set.
        If has_target=False:
            aug_x_nor : np.ndarray
                Normalized augmented features only.
        """
        # Apply the same augmentation pipeline as in training
        aug_data_df = self.augment_dataset(self.external_df)

        # Remove the first column (TIMESTAMP) before building the final arrays
        aug_data_df_no_time = aug_data_df.drop(columns=[aug_data_df.columns[0]])

        if self.has_target:
            # The last column is the target 'power'
            aug_x = aug_data_df_no_time.iloc[:, :-1].values
            aug_y = aug_data_df_no_time.iloc[:, -1].values

            # Normalize external augmented features using training statistics
            aug_x_nor = self.normalize_x(aug_x)

            return aug_x_nor, aug_y
        else:
            # If no target exists, use all columns as input features
            aug_x = aug_data_df_no_time.values
            aug_x_nor = self.normalize_x(aug_x)

            return aug_x_nor