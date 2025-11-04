"""
Results saving module for surrogate model experiments.
"""
import os
import csv


class SurrogateResultsSaver:
    """Handles saving surrogate experiment results to CSV files."""
    
    def __init__(self, csv_file_path='surrogate_results.csv'):
        self.csv_file_path = csv_file_path
        self.header = [
            'Model', 'Layer', 'Dataset', 'n_surrogate_dataset', 'n_train_dataset',
            'Test_MSE', 'Test_MAE', 'Test_SSIM', 'Test_Cosine_similarity',
            'Test_PSNR', 'Test_Pearson_correlation', 'Test_DTW', 'Test_QT', 'Test_QRS',
            'Test_Coherence',
            'Test_Min-max_norm_MSE', 'Test_Min-max_norm_MAE', 'Test_Min-max_norm_RMSE'
        ]
    
    def save_results(self, args, metrics):
        """
        Save surrogate experiment results to CSV file.
        
        Args:
            args: Experiment arguments
            metrics: Tuple of metric values in the order specified by header
        """
        data_row = [
            args.model, args.layer, args.dataset, args.surrogate_ratio, args.train_ratio, 
            *metrics
        ]
        
        file_exists = os.path.exists(self.csv_file_path)
        
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.header)
            writer.writerow(data_row)
            f.flush()
    
    def print_results(self, metrics):
        """Print formatted results to console."""
        metric_names = [
            'MSE', 'MAE', 'SSIM', 'Cosine similarity', 'PSNR',
            'Pearson correlation', 'DTW', 'QT', 'QRS', 'Coherence',
            'Min-max norm MSE', 'Min-max norm MAE', 'Min-max norm RMSE'
        ]
        
        result_str = "Test " + ", ".join([
            f"{name}: {value:.4f}"
            for name, value in zip(metric_names, metrics)
        ])
        
        print(result_str)
        print(f"Results successfully saved to '{self.csv_file_path}'.")