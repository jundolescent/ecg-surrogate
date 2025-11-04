"""
Data loading module for surrogate model experiments.
Extends the reconstruction data loader with surrogate-specific functionality.
"""
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from reconstruction.data_loader import ReconstructionDataLoader
from utils.utils import (
    resample_signals, normalize_zscore_per_sample, normalize_zscore_per_lead,
    normalize_minmax_per_sample, normalize_minmax_per_lead
)


class SurrogatePreprocessedDataset(Dataset):
    """Preprocessed dataset for surrogate experiments."""
    
    def __init__(self, original_dataset, args, device, label_type='train'):
        super().__init__()
        self.processed_x = []
        self.processed_inp = []
        self.processed_chan_matrix = []
        self.processed_time_matrix = []
        self.args = args
        self.device = device
        
        print(f"Starting full pre-processing of {label_type} dataset...")
        temp_loader = DataLoader(original_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        if self.args.model != 'HeartLang':
            for batch in tqdm(temp_loader):
                raw_ecg = batch['ecg']
                x, inp, L = self._process_input(raw_ecg, args, device)
                self.processed_x.append(x.cpu())
                self.processed_inp.append(inp.cpu())
        else:
            for batch in tqdm(temp_loader):
                x = batch[0]
                in_chan = batch[2]
                in_time = batch[3]
                inp = batch[4]
                x = normalize_zscore_per_lead(x)
                inp = normalize_zscore_per_lead(inp)
                _, L, _ = inp.shape
                self.processed_x.append(x.cpu())
                self.processed_inp.append(inp.cpu())
                self.processed_chan_matrix.append(in_chan.cpu())
                self.processed_time_matrix.append(in_time.cpu())
        
        self.length = L
        self.processed_x = torch.cat(self.processed_x, dim=0)
        self.processed_inp = torch.cat(self.processed_inp, dim=0)
        if self.args.model == 'HeartLang':
            self.processed_chan_matrix = torch.cat(self.processed_chan_matrix, dim=0)
            self.processed_time_matrix = torch.cat(self.processed_time_matrix, dim=0)
        print(f"Pre-processing complete. Total samples: {len(self.processed_x)}")
    
    def _process_input(self, x, args, device):
        """Process input based on model type."""
        if args.model == 'hubert-ecg':
            x = resample_signals(x, 500, 100)
            x = torch.tensor(x, dtype=torch.float32).to(device)
            x = normalize_minmax_per_lead(x)
            x = x[:, :, :100*5]
            temp = x
            _, _, L = x.shape
            x = x.reshape(x.shape[0], -1)
        elif args.model == 'ECGFounder':
            # x = normalize_zscore_per_lead(x)
            x = normalize_zscore_per_sample(x)
            _, _, L = x.shape
        elif args.model == 'ECG-FM':
            x = normalize_zscore_per_lead(x)
            x = x[:, :, :500*5]
            _, _, L = x.shape
        elif args.model == 'MERL':
            x = normalize_minmax_per_sample(x)
            x[:, [3, 4]] = x[:, [4, 3]]
            _, _, L = x.shape
        
        x = x.to(device)
        if args.model == 'hubert-ecg':
            inp = temp
        else:
            inp = x.clone()
            inp = inp.to(device)
        
        return x, inp, L
    
    def __len__(self):
        return len(self.processed_x)
    
    def __getitem__(self, idx):
        if self.args.model == 'HeartLang':
            return (self.processed_x[idx],
                    self.processed_chan_matrix[idx],
                    self.processed_time_matrix[idx],
                    self.processed_inp[idx])
        return self.processed_x[idx], self.processed_inp[idx]
    
    def get_length(self):
        return self.length


class SurrogateDataLoader(ReconstructionDataLoader):
    """Handles data loading for surrogate model experiments."""
    
    def __init__(self, args, device):
        super().__init__(args, device)
        self.surrogate_dataset = None
        self.surrogate_train_dataset = None
        self.surrogate_val_dataset = None
    
    def prepare_datasets(self):
        """Prepare datasets with surrogate-specific splits."""
        # First prepare base datasets
        if self.args.model == 'HeartLang':
            self._prepare_heartlang_datasets()
        elif self.args.dataset == 'harvard-emory':
            self._prepare_harvard_emory_surrogate_datasets()
        else:
            self._prepare_standard_datasets()
        
        # Apply surrogate-specific splits
        self._apply_surrogate_splits()
        
        # Preprocess all datasets
        self._preprocess_datasets()
    
    def _prepare_harvard_emory_surrogate_datasets(self):
        """Prepare Harvard-Emory datasets with surrogate splits."""
        from dataset.ecgdataset import ECGDataset
        
        train_dataset = ECGDataset(
            dataset_name=self.args.dataset,
            class_name=self.args.label_type,
            split='train'
        )
        
        # Split for surrogate experiments
        train_dataset, _ = random_split(train_dataset, [0.4, 0.6])
        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
        val_dataset, test_dataset = random_split(val_dataset, [0.5, 0.5])
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def _apply_surrogate_splits(self):
        """Apply surrogate-specific dataset splits."""
        if self.args.surrogate_ratio != 1.0:
            surrogate_ratio = int(self.args.surrogate_ratio)
            self.train_dataset, self.surrogate_dataset = random_split(
                self.train_dataset,
                [len(self.train_dataset) - surrogate_ratio, surrogate_ratio]
            )
            surrogate_len = len(self.surrogate_dataset)
            surrogate_train_size = int(surrogate_len * 0.9)
            surrogate_val_size = surrogate_len - surrogate_train_size
            self.surrogate_train_dataset, self.surrogate_val_dataset = random_split(
                self.surrogate_dataset,
                [surrogate_train_size, surrogate_val_size]
            )

        if self.args.train_ratio != 1.0:
            train_ratio = int(self.args.train_ratio)
            self.train_dataset, _ = random_split(
                self.train_dataset,
                [train_ratio, len(self.train_dataset) - train_ratio]
            )
    
    def _preprocess_datasets(self):
        """Apply preprocessing to all datasets including surrogate."""
        self.train_dataset = SurrogatePreprocessedDataset(
            self.train_dataset, self.args, self.device, 'train'
        )
        self.val_dataset = SurrogatePreprocessedDataset(
            self.val_dataset, self.args, self.device, 'val'
        )
        self.test_dataset = SurrogatePreprocessedDataset(
            self.test_dataset, self.args, self.device, 'test'
        )
        
        if self.surrogate_train_dataset is not None:
            self.surrogate_train_dataset = SurrogatePreprocessedDataset(
                self.surrogate_train_dataset, self.args, self.device, 'surrogate_train'
            )
            self.surrogate_val_dataset = SurrogatePreprocessedDataset(
                self.surrogate_val_dataset, self.args, self.device, 'surrogate_val'
            )
    
    def get_data_loaders(self):
        """Create and return all data loaders including surrogate loaders."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem
        )
        
        surrogate_loader = None
        surrogate_val_loader = None
        
        if self.surrogate_train_dataset is not None:
            surrogate_loader = DataLoader(
                self.surrogate_train_dataset,
                batch_size=self.args.surrogate_batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem
            )
            
            surrogate_val_loader = DataLoader(
                self.surrogate_val_dataset,
                batch_size=self.args.surrogate_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem
            )
        
        return train_loader, val_loader, test_loader, surrogate_loader, surrogate_val_loader