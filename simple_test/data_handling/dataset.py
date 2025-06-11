from torch.utils.data import Dataset 
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA

class EmbeddingsToAgeDataset(Dataset):
    def __init__(self, data_df, feature_names=[]):
        
        self.data = data_df

        # drop na columns with more than 30% missing values
        n = 0.3
        na_cols = self.data.columns[self.data.isna().mean() >  n]
        data_nona = self.data.drop(columns=na_cols)

        # drop rows with any na values
        data_nona = data_nona.dropna()

        # drop non float columns
        non_float_cols = data_nona.select_dtypes(exclude=[np.float64, np.int64]).columns
        data_nona = data_nona.drop(columns=non_float_cols)
        self.data = data_nona

        if len(feature_names) == 0:     # use all columns as input
            cols = self.data.columns.tolist()
            cols.remove('age')
            cols.remove('eid')
        else:                           # use specified feature names
            cols = feature_names

        self.X = self.data[cols].values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0) # normalize features
        self.y = self.data['age'].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
    
def create_datasets(data_path, feature_names=[], val_size=0.3):
    data = pd.read_csv(data_path)
    
    train, val = train_test_split(data, test_size=val_size, random_state=42)
    train_dataset = EmbeddingsToAgeDataset(train, feature_names=feature_names)
    val_dataset = EmbeddingsToAgeDataset(val, feature_names=feature_names)

    return train_dataset, val_dataset

def create_three_datasets(data_path, feature_names=[], val_size=0.3, test_size=0.2):
    data = pd.read_csv(data_path)
    train, temp = train_test_split(data, test_size=val_size + test_size, random_state=42)
    val, test = train_test_split(temp, test_size=test_size/(val_size + test_size), random_state=42)
    train_dataset = EmbeddingsToAgeDataset(train, feature_names=feature_names)
    val_dataset = EmbeddingsToAgeDataset(val, feature_names=feature_names)
    test_dataset = EmbeddingsToAgeDataset(test, feature_names=feature_names)
    #print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset
    