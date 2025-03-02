import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from config import *


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=CONTEXT_LENGTH, pred_len=OUT_LENGTH):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = np.arange(len(data[0]) - seq_len - pred_len + 1)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[:, i:i+self.seq_len]
        if self.pred_len == 1:
            y = self.data[:, i+1:i+self.seq_len+1]
        else:
            y = self.data[:, i+self.seq_len:i+self.seq_len+self.pred_len]
        return x, y


def load_and_preprocess_data(file_path, features=None, max_samples=None):
    df = pd.read_csv(file_path)
    
    if features is None:
        features = []
        if 'midprice' in df.columns:
            features.append('midprice')
        elif 'close' in df.columns:
            features.append('close')
            
        if 'volume' in df.columns:
            features.append('volume')
            
        if 'reward' in df.columns:
            features.append('reward')
    
    print(f"Using features: {features}")
    
    if max_samples is not None:
        data = df[features].values[-max_samples:]
    else:
        data = df[features].values

    data = np.asarray(data, dtype=np.float32)
    
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    
    return data, mean, std, features


def save_processed_data(data, mean, std, features, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        data=data,
        mean=mean,
        std=std,
        features=features
    )
    return save_path


def load_processed_data(save_path):
    data = np.load(save_path)
    return data['data'], data['mean'], data['std'], data['features']


def create_dataloaders(data, batch_size=BATCH_SIZE, train_split=0.8, val_split=0.1, seq_len=CONTEXT_LENGTH, pred_len=OUT_LENGTH):
    n = data.shape[1]
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    
    train_data = data[:, :train_size]
    val_data = data[:, train_size:train_size+val_size]
    test_data = data[:, train_size+val_size:]
    
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)
    
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
    
    data_norm = (data - mean) / (std + 1e-8)
    return data_norm, mean, std


def denormalize_data(data_norm, mean, std):
    return data_norm * std + mean


def load_dataset(file_path, features=None, max_samples=None, batch_size=BATCH_SIZE, 
                train_split=0.8, val_split=0.1, seq_len=CONTEXT_LENGTH, pred_len=OUT_LENGTH,
                use_cached=True):
    
    cache_dir = os.path.join(os.path.dirname(file_path), 'cache')
    cache_file = os.path.join(cache_dir, f"{os.path.basename(file_path)}.npz")
    
    if use_cached and os.path.exists(cache_file):
        print(f"Loading preprocessed data from cache: {cache_file}")
        raw_data, mean, std, features = load_processed_data(cache_file)
    else:
        raw_data, mean, std, features = load_and_preprocess_data(file_path, features, max_samples)
        if use_cached:
            print(f"Saving preprocessed data to cache: {cache_file}")
            save_processed_data(raw_data, mean, std, features, cache_file)
    
    data_norm, _, _ = normalize_data(raw_data, mean, std)
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_tensor,
        batch_size, train_split, val_split, seq_len, pred_len
    )
    
    dataset_info = {
        'raw_data': raw_data,
        'normalized_data': data_norm,
        'mean': mean,
        'std': std,
        'features': features
    }
    
    return train_loader, val_loader, test_loader, dataset_info