"""
Training module for surrogate model experiments.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.prepare_decoder import prepare_representation
from utils.metrics import cosine_similarity_loss
from .loss_function import mmd_channelwise_loss, mmd_global_loss
import random
import matplotlib.pyplot as plt
import numpy as np
from .utils import prepare_batch_data, forward_reconstruction
from .loss_function import GaussianKernel, MultipleKernelMaximumMeanDiscrepancy

class SurrogateTrainer:
    """Handles training of surrogate models."""
    
    def __init__(self, model, surrogate_model, args, device, activations):
        self.model = model
        self.surrogate_model = surrogate_model
        self.args = args
        self.device = device
        self.activations = activations
        
        # Setup training components
        self.optimizer = torch.optim.AdamW(
            surrogate_model.parameters(),
            lr=args.surrogate_lr,
            weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()
        
    def _prepare_batch_data(self, sample):
        """Prepare batch data based on model type."""
        return prepare_batch_data(sample, self.args.model, self.device)
    
    def _forward_pass(self, x):
        """Forward pass through the model and decoder."""
        return forward_reconstruction(
            self.model, self.recon_decoder, x, 
            self.args.model, self.args.layer, self.activations
        )
    
    def train_surrogate_epoch(self, surrogate_loader):
        """Train surrogate model for one epoch."""
        self.surrogate_model.train()
        total_loss = 0
        
        for sample in tqdm(surrogate_loader):
            x, inp = self._prepare_batch_data(sample)

            # Get target representation from original model
            z = prepare_representation(
                self.model, x, self.args.model, self.args.layer, self.activations
            )

            if self.args.model == 'hubert-ecg':
                x = x.unsqueeze(1)
            elif self.args.model == 'HeartLang':
                x = x[0]
            z_sur = self.surrogate_model(x)
            # Compute loss
            # loss = cosine_similarity_loss(z_sur, z)
            # print(z_sur.shape, z.shape)
            loss = self.criterion(z_sur, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(surrogate_loader)
    
    @torch.no_grad()
    def evaluate_surrogate(self, val_loader):
        """Evaluate surrogate model."""
        self.surrogate_model.eval()
        total_loss = 0
        
        for sample in tqdm(val_loader):
            x, inp = self._prepare_batch_data(sample)
            # Get target representation from original model
            z = prepare_representation(
                self.model, x, self.args.model, self.args.layer, self.activations
            )
            
            # # Get surrogate representation
            if self.args.model == 'hubert-ecg':
                x = x.unsqueeze(1)
            elif self.args.model == 'HeartLang':
                x = x[0]
            z_sur = self.surrogate_model(x)
            
            # Compute loss
            # loss = cosine_similarity_loss(z_sur, z)
            loss = self.criterion(z_sur, z)
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_surrogate(self, surrogate_loader, val_loader, n_epochs, patience):
        """Full surrogate training loop with early stopping."""
        best_loss = float('inf')
        patience_count = 0
        
        print(f"Training surrogate model start... {len(surrogate_loader)}")
        
        for epoch in range(1, n_epochs + 1):
            loss = self.train_surrogate_epoch(surrogate_loader)
            
            if epoch % 5 == 0 and epoch != n_epochs:
                val_loss = self.evaluate_surrogate(val_loader)
                print(f"[Epoch {epoch}] val loss: {val_loss:.4f}")
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


class SurrogateReconstructionTrainer:
    """Handles training of reconstruction decoder with surrogate model."""
    
    def __init__(self, surrogate_model, recon_decoder, args, device):
        self.surrogate_model = surrogate_model
        self.recon_decoder = recon_decoder
        self.args = args
        self.device = device
        
        # Setup training components
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            recon_decoder.parameters(),
            lr=args.lr,
            weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6
        )
        
    def _prepare_batch_data(self, sample):
        """Prepare batch data based on model type."""
        return prepare_batch_data(sample, self.args.model, self.device)
    
    def _forward_pass(self, x):
        """Forward pass through the model and decoder."""
        return forward_reconstruction(
            self.model, self.recon_decoder, x, 
            self.args.model, self.args.layer, self.activations
        )
    
    def train_reconstruction_epoch(self, train_loader):
        """Train reconstruction decoder for one epoch using surrogate model."""
        self.recon_decoder.train()
        total_loss = 0
        
        for sample in tqdm(train_loader):
            x, inp = self._prepare_batch_data(sample)
            if self.args.model == 'hubert-ecg':
                    x = x.unsqueeze(1)
            elif self.args.model == 'HeartLang':
                x = x[0]
            with torch.no_grad():
                z = self.surrogate_model(x)
            
            x_recon = self.recon_decoder(z)
            
            if self.args.model == 'hubert-ecg':
                x_recon = x_recon.reshape(x_recon.size(0), 12, -1)
            
            loss = self.criterion(x_recon, inp)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate_reconstruction(self, val_loader):
        """Evaluate reconstruction decoder using surrogate model."""
        self.recon_decoder.eval()
        self.surrogate_model.eval()
        total_mse = 0
        
        for sample in tqdm(val_loader):
            x, inp = self._prepare_batch_data(sample)
            if self.args.model == 'hubert-ecg':
                    x = x.unsqueeze(1)
            elif self.args.model == 'HeartLang':
                x = x[0]
            with torch.no_grad():
                z = self.surrogate_model(x)
                x_recon = self.recon_decoder(z)
            
            if self.args.model == 'hubert-ecg':
                x_recon = x_recon.reshape(x_recon.size(0), 12, -1)
            
            loss = self.criterion(x_recon, inp)
            total_mse += loss.item()
        
        return total_mse / len(val_loader)
    
    def train_reconstruction(self, train_loader, val_loader, n_epochs, patience):
        """Full reconstruction training loop with early stopping."""
        best_loss = float('inf')
        patience_count = 0
        
        print(f"Training decoder model with train dataset start... {len(train_loader)}")
        
        for epoch in range(1, n_epochs + 1):
            loss = self.train_reconstruction_epoch(train_loader)
            
            if epoch % 5 == 0 and epoch != n_epochs:
                self.recon_decoder.eval()
                mse_loss = self.evaluate_reconstruction(val_loader)
                print(f"[Epoch {epoch}] MSE: {mse_loss:.4f}")
                
                self.scheduler.step(mse_loss)
                
                if mse_loss < best_loss:
                    best_loss = mse_loss
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


class SurrogateFinetuner:
    """Handles fine-tuning of reconstruction decoder with surrogate dataset."""
    
    def __init__(self, model, recon_decoder, args, device, activations):
        self.model = model
        self.recon_decoder = recon_decoder
        self.args = args
        self.device = device
        self.activations = activations
        
        # Setup fine-tuning components
        self.criterion = nn.MSELoss()
        finetune_lr = args.lr
        self.optimizer = torch.optim.AdamW(
            recon_decoder.parameters(),
            lr=finetune_lr,
            weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        
    def _prepare_batch_data(self, sample):
        """Prepare batch data based on model type."""
        return prepare_batch_data(sample, self.args.model, self.device)
    
    def _forward_pass(self, x):
        """Forward pass through the model and decoder."""
        return forward_reconstruction(
            self.model, self.recon_decoder, x, 
            self.args.model, self.args.layer, self.activations
        )
    
    def finetune_epoch(self, surrogate_loader):
        """Fine-tune reconstruction decoder for one epoch."""
        self.recon_decoder.train()
        total_loss = 0
        
        for sample in tqdm(surrogate_loader):
            x, inp = self._prepare_batch_data(sample)
            with torch.no_grad():
                z = prepare_representation(
                    self.model, x, self.args.model, self.args.layer, self.activations
                )
            
            x_recon = self.recon_decoder(z)
            
            if self.args.model == 'hubert-ecg':
                x_recon = x_recon.reshape(x_recon.size(0), 12, -1)
            
            loss = self.criterion(x_recon, inp)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(surrogate_loader)
    
    @torch.no_grad()
    def evaluate_finetune(self, val_loader):
        """Evaluate fine-tuned reconstruction decoder."""
        self.recon_decoder.eval()
        total_mse = 0
        
        for sample in tqdm(val_loader):
            x, inp = self._prepare_batch_data(sample)
            with torch.no_grad():
                z = prepare_representation(
                    self.model, x, self.args.model, self.args.layer, self.activations
                )
                x_recon = self.recon_decoder(z)
            
            if self.args.model == 'hubert-ecg':
                x_recon = x_recon.reshape(x_recon.size(0), 12, -1)
            
            loss = self.criterion(x_recon, inp)
            total_mse += loss.item()
        
        return total_mse / len(val_loader)
    
    def finetune(self, surrogate_loader, val_loader, n_epochs=100, patience=5):
        """Full fine-tuning loop with early stopping."""
        best_loss = float('inf')
        patience_count = 0
        
        print(f"Training decoder model with surrogate dataset start... {len(surrogate_loader)}")
        
        for epoch in range(1, n_epochs + 1):
            loss = self.finetune_epoch(surrogate_loader)
            
            if epoch % 5 == 0 and epoch != n_epochs:
                mse_loss = self.evaluate_finetune(val_loader)
                self.scheduler.step(mse_loss)
                print(f"[Epoch {epoch}] MSE: {mse_loss:.4f}")
                
                if mse_loss < best_loss:
                    best_loss = mse_loss
                    patience_count = 0
                else:
                    patience_count += 1
                
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


class RepresentationsCollector:
    """Collects representations from the original model for surrogate training."""
    
    def __init__(self, model, args, device, activations):
        self.model = model
        self.args = args
        self.device = device
        self.activations = activations
        self.mean = 0.0
        self.std = 1.0

        self.train_x = None
        self.val_x = None
        self.train_representations = None
        self.val_representations = None
        
        self.val_split_ratio = getattr(args, 'val_split', 0.2)
        
    def collect_representations(self, surrogate_loader):
        """Collect representations from the original model."""
        print("Collecting only representations from surrogate dataset...")
        representations = []
        x_list = []
        for sample in tqdm(surrogate_loader):
            x, inp = sample
            x = x.to(self.device)
            inp = inp.to(self.device)
            with torch.no_grad():
                z = prepare_representation(
                    self.model, x, self.args.model, self.args.layer, self.activations
                )
                representations.append(z.cpu())
                x_list.append(x.cpu())

        all_x_list = torch.cat(x_list, dim=0)
        all_representations = torch.cat(representations, dim=0)
        total_size = all_representations.shape[0]
        val_size = int(total_size * self.val_split_ratio)
        train_size = total_size - val_size
        
        indices = torch.randperm(total_size)
        self.train_x = all_x_list[indices[:train_size]]
        self.val_x = all_x_list[indices[train_size:]]
        self.train_representations = all_representations[indices[:train_size]]
        self.val_representations = all_representations[indices[train_size:]]
        
    def get_representations(self):
        """Return collected representations."""
        return self.train_representations, self.val_representations

    def get_inputs(self):
        """Return collected inputs."""
        return self.train_x, self.val_x

    def save_representations(self, path):
        """Save collected representations."""
        torch.save({
            'train_representations': self.train_representations,
            'val_representations': self.val_representations
        }, path)
        
    def save_x_inputs(self, path):
        """Save collected inputs."""
        torch.save({
            'train_x': self.train_x,
            'val_x': self.val_x
        }, path)

# class FORAsurrogateTrainer:
#     """
#     MMD Loss만을 사용하여 Surrogate 모델(Generator)을 훈련시킵니다.
#     GAN 관련 코드 (D 모델, D 옵티마이저, D Loss, Adv Loss)는 모두 제거되었습니다.
#     """
#     def __init__(self, surrogate_model, args, device):
#         self.G = surrogate_model
#         # self.D = ... # D 모델 제거
        
#         self.args = args
#         self.device = device
        
#         # --- (1) Loss 함수 정의 ---
#         # self.criterion_stat = nn.MSELoss() # (Mean/Std용 Loss) - 사용하지 않음
        
#         # (2) 옵티마이저 (G용 1개만 필요)
#         self.optimizer_G = torch.optim.AdamW(
#             self.G.parameters(),
#             lr=args.surrogate_lr,
#             weight_decay=1e-2, betas=(0.5, 0.999)
#         )
#         # self.optimizer_D = ... # D 옵티마이저 제거
        
#         self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer_G, mode='min', factor=0.5, patience=1, min_lr=1e-6
#         )
#         # self.scheduler_D = ... # D 스케줄러 제거
        
#         self.lambda_stat = 1.0  # (MMD Loss 가중치, 메인 Loss이므로 1.0으로 설정)
#         # self.lambda_gp = ... # GP 가중치 제거
#         # self.D_CRITIC_ITERS = 5 # D 훈련 횟수 제거

#     def train_epoch(self, train_loader, train_representations):
#         """
#         MMD Loss만을 사용하여 G (Surrogate)를 훈련합니다.
#         """
        
#         z_real_dataset = torch.utils.data.TensorDataset(train_representations)
#         z_real_loader = torch.utils.data.DataLoader(z_real_dataset, 
#                                                      batch_size=train_loader.batch_size, 
#                                                      shuffle=True, 
#                                                      drop_last=True)
#         z_real_iter = iter(z_real_loader)
        
#         self.G.train()
        
#         total_loss_g = 0.0
#         repeat = 1 # G 훈련 횟수
        
#         # D 훈련 루프 제거
#         for batch_idx, sample in enumerate(tqdm(train_loader, desc=f"Training Surrogate (MMD Only)")): 
#             x_pub, inp = sample
#             x_pub = x_pub.to(self.device)
            
#             if self.args.model == 'hubert-ecg':
#                 x_pub = x_pub.unsqueeze(1)
            
#             # z_real 배치 로드 (G 훈련에만 사용)
#             try:
#                 z_real_sample = next(z_real_iter)
#             except StopIteration:
#                 z_real_iter = iter(z_real_loader)
#                 z_real_sample = next(z_real_iter)
            
#             z_real = z_real_sample[0].to(self.device)
            
#             # ==========================================================
#             # 1. Generator (G, 즉 Surrogate) 훈련 (MMD Loss)
#             # ==========================================================
            
#             current_batch_g_loss = 0.0
#             for _ in range(repeat):  # repeat=1
#                 self.optimizer_G.zero_grad()
                
#                 # (A) '가짜' z_sur 생성 
#                 z_sur_fake = self.G(x_pub)
                
#                 # (가) GAN Adv Loss 제거됨
#                 # loss_G_adv = -torch.mean(self.D(z_sur_fake)) 
                
#                 # (나) MMD Global Loss 계산
#                 min_batch = min(z_sur_fake.size(0), z_real.size(0))
                
#                 # MMD Loss는 평탄화된 텐서를 사용해야 함 (mmd_global_loss 가정)
#                 z_sur_flat = z_sur_fake[:min_batch].view(min_batch, -1)
#                 z_real_flat = z_real[:min_batch].view(min_batch, -1)
                
#                 # mmd_global_loss 함수가 평탄화된 텐서를 받도록 가정
#                 loss_G_stat = mmd_global_loss(
#                     z_sur_flat, z_real_flat, bandwidths=[0.1, 1.0, 10.0]
#                 )

#                 # 최종 G Loss: MMD Loss만 사용 (공격 Loss는 이 코드에 없으므로 MMD만)
#                 loss_G = self.lambda_stat * loss_G_stat 
                
#                 loss_G.backward()
#                 self.optimizer_G.step()
                
#                 current_batch_g_loss += loss_G.item()
                
#             total_loss_g += (current_batch_g_loss / repeat)
            
#         avg_loss_g = total_loss_g / len(train_loader)
        
#         # D Loss는 이제 출력하지 않습니다.
#         print(f"Epoch Avg Loss G (MMD): {avg_loss_g:.4f}")

#     @torch.no_grad()
#     def evaluate(self, val_loader, val_representations):
#         """
#         Evaluate surrogate model using MMD loss (이전 Mean/Std loss).
#         """
#         self.G.eval() # (G = surrogate_model)
#         total_loss = 0
#         z_real_dataset = torch.utils.data.TensorDataset(val_representations)
#         z_real_loader = torch.utils.data.DataLoader(z_real_dataset, 
#                                                      batch_size=val_loader.batch_size, 
#                                                      shuffle=True, 
#                                                      drop_last=True)
#         z_real_iter = iter(z_real_loader)
#         total_mean_diff = 0
#         total_cov_diff = 0
#         for sample in tqdm(val_loader, desc="Validating Surrogate (MMD)"):
#             x, inp = sample
#             x = x.to(self.device)

#             try:
#                 z_real_sample = next(z_real_iter)
#             except StopIteration:
#                 z_real_iter = iter(z_real_loader)
#                 z_real_sample = next(z_real_iter)

#             z_real = z_real_sample[0].to(self.device)

#             if self.args.model == 'hubert-ecg':
#                 x = x.unsqueeze(1)

#             z_sur = self.G(x) # (G = surrogate_model)
#             min_batch = min(z_sur.size(0), z_real.size(0))
            
#             # MMD Loss는 평탄화된 텐서를 사용해야 함 (mmd_global_loss 가정)
#             z_sur_flat = z_sur[:min_batch].view(min_batch, -1)
#             z_real_flat = z_real[:min_batch].view(min_batch, -1)
            
#             loss = mmd_global_loss(
#                 z_sur_flat, z_real_flat, bandwidths=[0.1, 1.0, 10.0]
#             )
#             mean_diff, cov_diff = self.calculate_mean_and_cov_diff(z_sur_flat, z_real_flat)
            
#             total_mean_diff += mean_diff
#             total_cov_diff += cov_diff
#             total_loss += loss.item()
            
#             avg_mean_diff = total_mean_diff / x[0].size(0)
#             avg_cov_diff = total_cov_diff / x[0].size(0)
        
#         avg_mean_diff = total_mean_diff / len(val_loader)
#         avg_cov_diff = total_cov_diff / len(val_loader)
#         print(f"Avg Mean Diff: {avg_mean_diff:.4f}, Avg Cov Diff: {avg_cov_diff:.4f}")
        
#         return total_loss / len(val_loader)

#     def calculate_mean_and_cov_diff(self, z_sur, z_real):
#         """
#         Surrogate Feature와 Real Feature의 Mean, Covariance 차이를 계산합니다.
#         (z_sur, z_real은 [Batch_Size, Feature_Dim] 형태를 가정)
#         """
#         # 1. Mean Difference (L2 norm)
#         mean_sur = torch.mean(z_sur, dim=0)
#         mean_real = torch.mean(z_real, dim=0)
#         mean_diff = torch.linalg.norm(mean_sur - mean_real, ord=2)

#         # 2. Covariance Difference (Frobenius norm)
#         # 배치 크기 N
#         N = z_sur.size(0)
        
#         # Feature를 Zero-mean으로 만듭니다.
#         z_sur_zm = z_sur - mean_sur
#         z_real_zm = z_real - mean_real

#         # Covariance Matrix 계산 (C = (1/(N-1)) * Z_zm.T @ Z_zm)
#         # torch.cov는 배치 차원을 지원하지 않으므로 직접 계산
#         cov_sur = (z_sur_zm.T @ z_sur_zm) / (N - 1)
#         cov_real = (z_real_zm.T @ z_real_zm) / (N - 1)

#         # Covariance Difference (Frobenius norm)
#         cov_diff = torch.linalg.norm(cov_sur - cov_real, ord='fro')

#         return mean_diff.item(), cov_diff.item()

#     def train_surrogate(self, train_loader, val_loader, n_epochs, patience, train_representations, val_representations):
#         """Full surrogate training loop with early stopping."""
#         best_loss = float('inf')
#         patience_count = 0
        
#         print(f"Training FORA surrogate model (MMD Only) start... {len(train_loader)}")
        
#         for epoch in range(1, n_epochs + 1):
#             self.train_epoch(train_loader, train_representations)
            
#             if epoch % 5 == 0 or epoch == n_epochs or epoch == 1:
#                 val_loss = self.evaluate(val_loader, val_representations)
#                 print(f"[Epoch {epoch}] val loss (MMD): {val_loss:.4f}")
#                 self.plot_feature_histograms(val_loader, val_representations, num_dims=5, epoch=epoch)
#                 self.scheduler_G.step(val_loss)
                
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     patience_count = 0
#                     self.save_model(epoch)
#                 else:
#                     patience_count += 1
                
#                 if patience_count >= patience:
#                     print(f"Early stopping at epoch {epoch}")
#                     break
                
#     @torch.no_grad()
#     def plot_feature_histograms(self, val_loader, val_representations, num_dims=5, epoch=None):
#         """
#         Validation set의 Real Feature와 Surrogate Feature를 샘플링하여 
#         랜덤으로 선택된 차원들의 Histogram을 비교하여 저장합니다.
        
#         Args:
#             num_dims (int): 시각화할 대표 Feature 차원의 개수.
#             epoch (int, optional): 파일명에 사용될 에폭 번호.
#         """
#         self.G.eval()
        
#         # 1. Real Feature와 Surrogate Feature 전체를 수집
#         all_z_real = []
#         all_z_sur = []

#         z_real_dataset = torch.utils.data.TensorDataset(val_representations)
#         z_real_loader = torch.utils.data.DataLoader(z_real_dataset, 
#                                                     batch_size=val_loader.batch_size, 
#                                                     shuffle=True, 
#                                                     drop_last=True)
#         z_real_iter = iter(z_real_loader)
        
#         # 전체 Validation 데이터셋을 순회하며 Feature를 수집합니다.
#         for sample in tqdm(val_loader, desc="Collecting Features for Histogram"):
#             x, _ = sample
#             x = x.to(self.device)

#             try:
#                 z_real_sample = next(z_real_iter)
#             except StopIteration:
#                 # Z_real 로더가 Z_pub 로더보다 작을 수 있으므로 재순환
#                 z_real_iter = iter(z_real_loader)
#                 z_real_sample = next(z_real_iter)

#             z_real = z_real_sample[0].to(self.device)

#             if self.args.model == 'hubert-ecg':
#                 x = x.unsqueeze(1)

#             z_sur = self.G(x)
#             min_batch = min(z_sur.size(0), z_real.size(0))
            
#             # MMD에서 사용했던 것처럼 Feature를 평탄화 (Feature Vector)
#             z_sur_flat = z_sur[:min_batch].view(min_batch, -1)
#             z_real_flat = z_real[:min_batch].view(min_batch, -1)
            
#             all_z_sur.append(z_sur_flat.cpu())
#             all_z_real.append(z_real_flat.cpu())
            
#         all_z_sur = torch.cat(all_z_sur, dim=0).numpy()
#         all_z_real = torch.cat(all_z_real, dim=0).numpy()
        
#         # 전체 Feature 차원 개수
#         total_feature_dims = all_z_sur.shape[1]
        
#         # 2. 대표 차원 선택
#         # 시각화할 num_dims 개의 차원을 랜덤하게 선택
#         selected_dims = np.random.choice(total_feature_dims, size=num_dims, replace=False)
        
#         # 3. Histogram 그리기 및 저장
#         fig, axes = plt.subplots(1, num_dims, figsize=(4 * num_dims, 4))
#         if num_dims == 1: # 1개 차원만 그릴 경우 axes가 1차원 배열이 아니므로 처리
#             axes = [axes]
            
#         for i, dim_idx in enumerate(selected_dims):
#             ax = axes[i]
            
#             # Real Feature (파란색)
#             ax.hist(all_z_real[:, dim_idx], bins=50, alpha=0.6, label='Real Feature', color='blue', density=True)
#             # Surrogate Feature (주황색)
#             ax.hist(all_z_sur[:, dim_idx], bins=50, alpha=0.6, label='Surrogate Feature', color='orange', density=True)
            
#             ax.set_title(f'Dimension {dim_idx} Distribution')
#             ax.legend()
            
#         plt.tight_layout()
        
#         # 파일 저장 경로 설정 (에폭 정보 포함)
#         epoch_str = f"_epoch{epoch}" if epoch is not None else ""
#         filename = f'./figures/fora/hist_comp_{self.args.model}_{self.args.layer}_{self.args.dataset}{epoch_str}.png'
        
#         # 디렉토리가 없으면 생성 (필요에 따라 구현)
#         # import os; os.makedirs(os.path.dirname(filename), exist_ok=True)
        
#         plt.savefig(filename)
#         plt.close(fig)
#         print(f"Histogram comparison saved to {filename}")                
                
#     def save_model(self, epochs):
#         """Save the trained surrogate model."""
#         torch.save(self.G.state_dict(), f'./surrogate/ckpts/surrogate_{self.args.model}_{self.args.layer}_{self.args.dataset}_{self.args.surrogate_ratio}_{self.args.train_ratio}_MMD.pth')