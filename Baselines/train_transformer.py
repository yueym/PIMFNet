import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import time
import gc
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
import math


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== MultiTaskHead（与LTC结构对齐）====================
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim=256, dropout_rate=0.15):  # Transformer适配的dropout
        super(MultiTaskHead, self).__init__()
        print(f"[INFO] MultiTaskHead initialized with input_dim={input_dim} (Transformer Baseline)")

        # 网络结构与LTC一致
        self.shared = nn.Sequential(
            nn.Conv2d(input_dim, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(384, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.point_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.prob_mu = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.prob_log_sigma = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.interval_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(128, 4, kernel_size=1),
            nn.Softplus()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        log_sigma_convs = [m for m in self.prob_log_sigma.modules() if isinstance(m, nn.Conv2d)]
        if len(log_sigma_convs) > 0:
            last_conv = log_sigma_convs[-1]
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, -2.5)

    def forward(self, x):
        shared_feat = self.shared(x)

        point_pred = self.point_head(shared_feat).squeeze(1)
        mu = self.prob_mu(shared_feat).squeeze(1)
        log_sigma = self.prob_log_sigma(shared_feat).squeeze(1)
        log_sigma = torch.clamp(log_sigma, min=-4.6, max=-1.2)
        sigma = torch.exp(log_sigma)

        interval_deltas = self.interval_head(shared_feat)
        interval_cumsum = torch.cumsum(interval_deltas, dim=1)
        interval_max = interval_cumsum[:, -1:, :, :] + 1e-6
        interval_sorted = interval_cumsum / interval_max
        interval_sorted = torch.clamp(interval_sorted, min=0.0, max=1.0)

        return point_pred, mu, sigma, interval_sorted


# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    """可学习的位置编码 + 正弦位置编码混合"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 正弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

        # 可学习的位置编码
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 混合正弦和可学习位置编码
        pos_encoding = self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        x = x + pos_encoding
        return self.dropout(x)


# ==================== Transformer模型（与LTC架构对齐）====================
class TransformerWindSpeedPredictor(nn.Module):
    """
    Transformer风速预测模型 - 与LTC架构对齐版本
    设计要点：
    1. 只使用时序分支（与LTC一致）
    2. 使用Transformer Encoder处理时序
    3. 添加output_layer（与LTC一致）
    4. 数据流与LTC完全一致
    """

    def __init__(self, H, W, tigge_features=10, dropout_rate=0.15,
                 d_model=256, nhead=8, num_layers=3, dim_feedforward=512):
        super(TransformerWindSpeedPredictor, self).__init__()
        print("[INFO] TransformerWindSpeedPredictor initialized - TEMPORAL-ONLY (Aligned with LTC architecture)")

        self.H = H
        self.W = W
        self.d_model = d_model

        # ==================== 输入投影层 ====================
        # 将输入特征维度(15)投影到d_model维度
        input_dim = tigge_features + 5  # 10 + 5 = 15
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # ==================== 位置编码 ====================
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout_rate)

        # ==================== Transformer Encoder ====================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='gelu',  # GELU激活函数，Transformer标准配置
            batch_first=True,
            norm_first=True  # Pre-LN，训练更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # ==================== 时序聚合层 ====================
        # 使用注意力池化聚合序列
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # ==================== output_layer（与LTC一致）====================
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_model // 2, d_model)
        )

        # ==================== MultiTaskHead ====================
        self.multi_task_head = MultiTaskHead(
            input_dim=d_model,
            dropout_rate=dropout_rate
        )

        # 权重初始化
        self._initialize_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Transformer Model Total parameters: {total_params:,}")
        print(f"[INFO] Transformer Model Trainable parameters: {trainable_params:,}")

    def _initialize_weights(self):
        # 输入投影层初始化
        for m in self.input_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # output_layer初始化
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 注意力池化初始化
        for m in self.attention_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):
        """
        forward流程与LTC对齐，只使用时序分支

        输入（保持与原Dataset兼容）：
            tigge_spatial: (B, 10, H, W) - 不使用
            dem_spatial: (B, 3, H, W) - 不使用
            interaction_spatial: (B, 5, H, W) - 不使用
            tigge_seq: (B, seq_len, 10, H, W) - 使用
            time_features_t: (B, 5) - 不使用
            time_features_seq: (B, seq_len, 5) - 使用
        """
        B, seq_len, C, H, W = tigge_seq.shape

        # ==================== 数据处理流程与LTC一致 ====================
        # tigge_seq: (B, seq_len, 10, H, W) -> (B*H*W, seq_len, 10)
        tigge_seq_reshaped = tigge_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, seq_len, C)

        # time_features_seq: (B, seq_len, 5) -> (B*H*W, seq_len, 5)
        time_seq_expanded = time_features_seq.unsqueeze(1).unsqueeze(2).expand(B, H, W, seq_len, 5)
        time_seq_expanded = time_seq_expanded.reshape(B * H * W, seq_len, 5)

        # 拼接TIGGE和时间特征：(B*H*W, seq_len, 15)
        transformer_input = torch.cat([tigge_seq_reshaped, time_seq_expanded], dim=-1)

        # ==================== 输入投影 ====================
        # (B*H*W, seq_len, 15) -> (B*H*W, seq_len, d_model)
        x = self.input_projection(transformer_input)

        # ==================== 位置编码 ====================
        x = self.pos_encoder(x)

        # ==================== Transformer Encoder ====================
        # (B*H*W, seq_len, d_model) -> (B*H*W, seq_len, d_model)
        x = self.transformer_encoder(x)

        # ==================== 时序聚合 ====================
        # 使用注意力权重加权求和
        attn_weights = self.attention_pool(x)  # (B*H*W, seq_len, 1)
        h = torch.sum(x * attn_weights, dim=1)  # (B*H*W, d_model)

        # ==================== 通过output_layer（与LTC一致） ====================
        h = torch.clamp(h, -100, 100)  # 与LTC一致的clamp
        out = self.output_layer(h)  # (B*H*W, d_model)

        # 重塑回空间维度：(B*H*W, d_model) -> (B, d_model, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, d_model, H, W)

        # ==================== MultiTaskHead输出 ====================
        point_pred, mu, sigma, interval_pred = self.multi_task_head(out)

        return point_pred, mu, sigma, interval_pred


# ==================== WindDataset（保持不变） ====================
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset variables: {list(self.ds.data_vars.keys())}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset dimensions: {dict(self.ds.dims)}")

        assert 'tigge_features' in self.ds.data_vars
        assert 'dem_features' in self.ds.data_vars
        assert 'interaction_features' in self.ds.data_vars
        assert 'time_features' in self.ds.data_vars
        assert 'target' in self.ds.data_vars

        tigge_min = float(self.ds['tigge_features'].min().values)
        tigge_max = float(self.ds['tigge_features'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tigge_features range: [{tigge_min:.4f}, {tigge_max:.4f}]")

        dem_min = float(self.ds['dem_features'].min().values)
        dem_max = float(self.ds['dem_features'].max().values)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dem_features range: [{dem_min:.4f}, {dem_max:.4f}]")

        interaction_min = float(self.ds['interaction_features'].min().values)
        interaction_max = float(self.ds['interaction_features'].max().values)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] interaction_features range: [{interaction_min:.4f}, {interaction_max:.4f}]")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        time_data = self.ds['time_features'].values
        self.time_scaler = StandardScaler()
        normalized_time = self.time_scaler.fit_transform(time_data)

        self.ds['time_features_normalized'] = xr.DataArray(
            normalized_time,
            dims=self.ds['time_features'].dims,
            coords={'sample': self.ds['time_features'].coords['sample']}
        )

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reconstructing time-space index")
        times = pd.to_datetime({
            'year': time_data[:, 0].astype(int),
            'month': time_data[:, 1].astype(int),
            'day': time_data[:, 2].astype(int),
            'hour': time_data[:, 3].astype(int)
        })

        self.ds = self.ds.assign_coords(time=("sample", times)).sortby('time')
        self.time_points = np.unique(self.ds.time.values)
        self.T = len(self.time_points)
        self.samples_per_time = H * W
        self.sample_indices = np.arange(self.T - self.seq_len + 1)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total time points: {self.T}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Samples per time: {self.samples_per_time}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Valid samples (seq_len={seq_len}): {len(self.sample_indices)}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        actual_idx = self.sample_indices[idx]
        t = actual_idx + self.seq_len - 1
        seq_times = self.time_points[t - self.seq_len + 1: t + 1]

        tigge_seq_data = []
        time_features_seq = []

        for time_point in seq_times:
            mask = self.ds.time == time_point
            tigge_data = self.ds['tigge_features'].sel(sample=mask).values.reshape(self.H, self.W, -1)
            tigge_seq_data.append(tigge_data)
            time_feat = self.ds['time_features_normalized'].sel(sample=mask).values[0]
            time_features_seq.append(time_feat)

        tigge_seq = np.stack(tigge_seq_data)
        time_features_seq = np.stack(time_features_seq)

        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t

        tigge_spatial = self.ds['tigge_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        dem_spatial = self.ds['dem_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        interaction_spatial = self.ds['interaction_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        target = self.ds['target'].sel(sample=mask_t).values.reshape(self.H, self.W)

        time_features_t = time_features_seq[-1]

        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'interaction_spatial': torch.from_numpy(interaction_spatial).float().permute(2, 0, 1),
            'tigge_seq': torch.from_numpy(tigge_seq).float().permute(0, 3, 1, 2),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'time_features_seq': torch.from_numpy(time_features_seq).float(),
            'target': torch.from_numpy(target).float()
        }


# ==================== CRPS计算（保持不变） ====================
def compute_crps(mu, sigma, target):
    sigma = torch.clamp(sigma, min=0.01, max=0.3)
    z = (target - mu) / sigma
    z = torch.clamp(z, min=-5.0, max=5.0)
    phi_z = 0.5 * (1.0 + torch.erf(z / np.sqrt(2)))
    log_pdf = -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi)
    pdf_z = torch.exp(log_pdf)
    term1 = z * (2 * phi_z - 1)
    term2 = 2 * pdf_z
    term3 = 1.0 / np.sqrt(np.pi)
    crps = sigma * (term1 + term2 - term3)
    crps_mean = torch.abs(crps).mean()

    if torch.isnan(crps_mean) or torch.isinf(crps_mean):
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)
    if crps_mean > 1.0:
        crps_mean = torch.clamp(crps_mean, max=0.5)

    return crps_mean


# ==================== 区间指标计算（保持不变） ====================
def compute_interval_metrics(interval_pred, target, point_pred):
    interval_pred = interval_pred.float()
    target = target.float()
    point_pred = point_pred.float()

    q_0025 = interval_pred[:, 0, :, :]
    q_025 = interval_pred[:, 1, :, :]
    q_075 = interval_pred[:, 2, :, :]
    q_0975 = interval_pred[:, 3, :, :]

    q_0025_flat = q_0025.flatten()
    q_025_flat = q_025.flatten()
    q_075_flat = q_075.flatten()
    q_0975_flat = q_0975.flatten()
    target_flat = target.flatten()
    point_pred_flat = point_pred.flatten()

    coverage_95 = ((target_flat >= q_0025_flat) & (target_flat <= q_0975_flat)).float().mean().item()
    coverage_50 = ((target_flat >= q_025_flat) & (target_flat <= q_075_flat)).float().mean().item()

    point_pred_safe = torch.clamp(point_pred_flat.abs(), min=1e-3)
    width_95 = q_0975_flat - q_0025_flat
    mwp_95 = (width_95 / point_pred_safe).mean().item()
    width_50 = q_075_flat - q_025_flat
    mwp_50 = (width_50 / point_pred_safe).mean().item()

    mc_95 = mwp_95 / max(coverage_95, 1e-6)
    mc_50 = mwp_50 / max(coverage_50, 1e-6)

    # 添加异常值检测（调试用）
    if mwp_95 > 10 or mwp_50 > 10:
        near_zero_ratio = (point_pred_flat.abs() < 0.01).float().mean().item()
        if near_zero_ratio > 0.1:
            print(f"[WARNING] {near_zero_ratio * 100:.1f}% 的点预测接近零")

    return {
        'CP_95': coverage_95,
        'CP_50': coverage_50,
        'MWP_95': mwp_95,
        'MWP_50': mwp_50,
        'MC_95': mc_95,
        'MC_50': mc_50
    }


# ==================== 多任务损失（Transformer适配的权重） ====================
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.4, beta=0.28, gamma=0.30):  # Transformer适配的权重
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.point_criterion = nn.SmoothL1Loss()
        print(f"[INFO] MultiTaskLoss权重 (Transformer Baseline): α={alpha}, β={beta}, γ={gamma}")

    def forward(self, point_pred, mu, sigma, interval_pred, target):
        loss_point = self.point_criterion(point_pred, target)

        if torch.isnan(loss_point) or torch.isinf(loss_point):
            loss_point = torch.tensor(0.0, device=target.device, dtype=target.dtype)

        loss_crps = compute_crps(mu, sigma, target)

        if torch.isnan(loss_crps) or torch.isinf(loss_crps) or loss_crps > 1.0:
            loss_crps = torch.clamp(loss_crps, min=0.0, max=0.1)

        quantiles = torch.tensor([0.025, 0.25, 0.75, 0.975], device=target.device, dtype=target.dtype)
        loss_interval = 0.0
        for i, q in enumerate(quantiles):
            pred_q = interval_pred[:, i, :, :]
            error = target - pred_q
            loss_q = torch.maximum(q * error, (q - 1) * error)
            loss_interval += loss_q.mean()
        loss_interval = loss_interval / len(quantiles)

        if torch.isnan(loss_interval) or torch.isinf(loss_interval):
            loss_interval = torch.tensor(0.0, device=target.device, dtype=target.dtype)

        total_loss = (self.alpha * loss_point + self.beta * loss_crps + self.gamma * loss_interval)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = loss_point

        return total_loss, {
            'point_loss': loss_point.item(),
            'crps_loss': loss_crps.item(),
            'interval_loss': loss_interval.item(),
            'total_loss': total_loss.item()
        }


# ==================== Warmup学习率调度器（Transformer专用）====================
class WarmupCosineScheduler:
    """Warmup + Cosine Annealing学习率调度"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增加
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * lr_scale)

        return self.optimizer.param_groups[0]['lr']


# ==================== 可视化函数 ====================
def visualize_predictions(model, dataloader, device, save_dir='visualizations_transformer_task2', num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break

            tigge_spatial = batch['tigge_spatial'].to(device)
            dem_spatial = batch['dem_spatial'].to(device)
            interaction_spatial = batch['interaction_spatial'].to(device)
            tigge_seq = batch['tigge_seq'].to(device)
            time_features_t = batch['time_features_t'].to(device)
            time_features_seq = batch['time_features_seq'].to(device)
            target = batch['target'].to(device)

            point_pred, mu, sigma, interval_pred = model(
                tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq
            )

            h_center, w_center = 24, 48
            target_val = target[0, h_center, w_center].cpu().numpy()
            point_val = point_pred[0, h_center, w_center].cpu().numpy()
            mu_val = mu[0, h_center, w_center].cpu().numpy()
            sigma_val = sigma[0, h_center, w_center].cpu().numpy()
            q_vals = interval_pred[0, :, h_center, w_center].cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            x_range = np.linspace(max(0, mu_val - 4 * sigma_val), min(1, mu_val + 4 * sigma_val), 200)
            pdf = stats.norm.pdf(x_range, mu_val, sigma_val)

            axes[0].plot(x_range, pdf, 'b-', linewidth=2, label='Predicted PDF')
            axes[0].axvline(target_val, color='r', linestyle='--', linewidth=2, label=f'True: {target_val:.3f}')
            axes[0].axvline(mu_val, color='g', linestyle='--', linewidth=2, label=f'Mean: {mu_val:.3f}')
            axes[0].fill_between(x_range, 0, pdf, alpha=0.3)
            axes[0].set_xlabel('Wind Speed (normalized)', fontsize=12)
            axes[0].set_ylabel('Probability Density', fontsize=12)
            axes[0].set_title(f'Probabilistic Prediction (Sample {batch_idx + 1})', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            x_pos = [0]
            axes[1].errorbar(x_pos, [point_val],
                             yerr=[[point_val - q_vals[0]], [q_vals[3] - point_val]],
                             fmt='o', markersize=8, capsize=10, capthick=2,
                             label='95% CI', color='blue', linewidth=2)
            axes[1].errorbar(x_pos, [point_val],
                             yerr=[[point_val - q_vals[1]], [q_vals[2] - point_val]],
                             fmt='o', markersize=6, capsize=8, capthick=2,
                             label='50% CI (IQR)', color='green', linewidth=2)
            axes[1].scatter(x_pos, [target_val], color='red', s=100, marker='*', label=f'True: {target_val:.3f}',
                            zorder=5)
            axes[1].scatter(x_pos, [point_val], color='blue', s=80, marker='o', label=f'Point: {point_val:.3f}',
                            zorder=4)

            axes[1].set_xlim(-0.5, 0.5)
            axes[1].set_ylim(max(0, q_vals[0] - 0.1), min(1, q_vals[3] + 0.1))
            axes[1].set_xticks([])
            axes[1].set_ylabel('Wind Speed (normalized)', fontsize=12)
            axes[1].set_title(f'Interval Prediction (Sample {batch_idx + 1})', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/prediction_sample_{batch_idx + 1}.png', dpi=300, bbox_inches='tight')
            plt.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Visualizations saved to {save_dir}/")


# ==================== 训练函数（Transformer适配的超参数） ====================
def train_model(model, train_loader, val_loader, device,
                epochs=10, learning_rate=0.0002, weight_decay=0.00008,  # Transformer需要更小学习率
                patience=3, accumulation_steps=4, warmup_epochs=2):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training process (Transformer Baseline - Aligned with LTC)")

    criterion = MultiTaskLoss(alpha=1.4, beta=0.28, gamma=0.30)  # Transformer适配

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),  # Transformer标准配置
        eps=1e-8
    )

    # 使用Warmup + Cosine学习率调度
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, min_lr=1e-6)
    scaler = GradScaler()
    best_val_loss = float('inf')

    train_losses, val_losses = [], []
    train_mae, train_rmse, train_r2 = [], [], []
    val_mae, val_rmse, val_r2 = [], [], []
    train_crps, val_crps = [], []
    train_cp_95, train_cp_50, val_cp_95, val_cp_50 = [], [], [], []
    train_mwp_95, train_mwp_50, val_mwp_95, val_mwp_50 = [], [], [], []
    train_mc_95, train_mc_50, val_mc_95, val_mc_50 = [], [], [], []

    patience_counter = 0
    os.makedirs('checkpoints_transformer_task2', exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint directory created: checkpoints_transformer_task2")

    first_batch = next(iter(train_loader))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_seq shape: {first_batch['tigge_seq'].shape}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_seq shape: {first_batch['time_features_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()

        # 更新学习率
        current_lr = scheduler.step(epoch)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f}")

        model.train()
        train_loss = 0.0
        batch_count = 0

        train_point_preds, train_mus, train_sigmas, train_interval_preds, train_targets = [], [], [], [], []
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            try:
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                interaction_spatial = batch['interaction_spatial'].to(device)
                tigge_seq = batch['tigge_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                with autocast():
                    point_pred, mu, sigma, interval_pred = model(
                        tigge_spatial, dem_spatial, interaction_spatial,
                        tigge_seq, time_features_t, time_features_seq
                    )
                    loss, loss_dict = criterion(point_pred, mu, sigma, interval_pred, target)

                train_point_preds.append(point_pred.detach().cpu().float())
                train_mus.append(mu.detach().cpu().float())
                train_sigmas.append(sigma.detach().cpu().float())
                train_interval_preds.append(interval_pred.detach().cpu().float())
                train_targets.append(target.detach().cpu().float())

                scaler.scale(loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    if np.isnan(total_norm) or np.isinf(total_norm):
                        print(f"[CRITICAL] 梯度为NaN/Inf! 跳过batch {batch_idx}")
                        optimizer.zero_grad()
                        continue

                    if total_norm > 50.0:
                        print(f"[WARNING] 梯度过大: {total_norm:.2f}, 跳过batch {batch_idx}")
                        optimizer.zero_grad()
                        continue

                    # Transformer使用稍小的梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    if batch_idx % 100 == 0:
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gradient norm: {grad_norm:.4f} (before clip: {total_norm:.4f})")

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * tigge_spatial.size(0)
                batch_count += 1

                if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                    batch_time = time.time() - batch_start
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    eta = (len(train_loader) - batch_idx - 1) * batch_time / 60
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1}/{epochs}, "
                          f"Batch {batch_idx + 1}/{len(train_loader)} ({progress:.1f}%), "
                          f"Loss: {loss.item():.6f} [Point: {loss_dict['point_loss']:.4f}, "
                          f"CRPS: {loss_dict['crps_loss']:.4f}, Interval: {loss_dict['interval_loss']:.4f}], "
                          f"Batch time: {batch_time:.2f}s, ETA: {eta:.2f} min")
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing batch {batch_idx}: {str(e)}")
                continue

        train_loss = train_loss / len(train_loader.dataset) if batch_count > 0 else float('inf')
        train_losses.append(train_loss)

        # 计算训练指标
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing training metrics")
        train_point_preds_tensor = torch.cat(train_point_preds)
        train_mus_tensor = torch.cat(train_mus)
        train_sigmas_tensor = torch.cat(train_sigmas)
        train_interval_preds_tensor = torch.cat(train_interval_preds)
        train_targets_tensor = torch.cat(train_targets)

        train_point_preds_flat = train_point_preds_tensor.numpy().flatten()
        train_mus_flat = train_mus_tensor.numpy().flatten()
        train_sigmas_flat = train_sigmas_tensor.numpy().flatten()
        train_targets_flat = train_targets_tensor.numpy().flatten()

        train_mae_val = np.mean(np.abs(train_point_preds_flat - train_targets_flat))
        train_rmse_val = np.sqrt(np.mean((train_point_preds_flat - train_targets_flat) ** 2))
        train_mean = np.mean(train_targets_flat)
        ss_tot = np.sum((train_targets_flat - train_mean) ** 2)
        ss_res = np.sum((train_targets_flat - train_point_preds_flat) ** 2)
        train_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        train_crps_val = compute_crps(
            torch.from_numpy(train_mus_flat),
            torch.from_numpy(train_sigmas_flat),
            torch.from_numpy(train_targets_flat)
        ).item()

        train_interval_metrics = compute_interval_metrics(
            train_interval_preds_tensor,
            train_targets_tensor,
            train_point_preds_tensor
        )

        train_mae.append(train_mae_val)
        train_rmse.append(train_rmse_val)
        train_r2.append(train_r2_val)
        train_crps.append(train_crps_val)
        train_cp_95.append(train_interval_metrics['CP_95'])
        train_cp_50.append(train_interval_metrics['CP_50'])
        train_mwp_95.append(train_interval_metrics['MWP_95'])
        train_mwp_50.append(train_interval_metrics['MWP_50'])
        train_mc_95.append(train_interval_metrics['MC_95'])
        train_mc_50.append(train_interval_metrics['MC_50'])

        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Epoch {epoch + 1}/{epochs} Training Metrics:')
        print(
            f'  Loss: {train_loss:.6f} | MAE: {train_mae_val:.4f} | RMSE: {train_rmse_val:.4f} | R²: {train_r2_val:.4f}')
        print(f'  CRPS: {train_crps_val:.4f}')
        print(f'  CP_95: {train_interval_metrics["CP_95"]:.4f} | CP_50: {train_interval_metrics["CP_50"]:.4f}')
        print(f'  MWP_95: {train_interval_metrics["MWP_95"]:.4f} | MWP_50: {train_interval_metrics["MWP_50"]:.4f}')
        print(f'  MC_95: {train_interval_metrics["MC_95"]:.4f} | MC_50: {train_interval_metrics["MC_50"]:.4f}')

        # 验证阶段
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation")
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        val_point_preds, val_mus, val_sigmas, val_interval_preds, val_targets = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    tigge_spatial = batch['tigge_spatial'].to(device)
                    dem_spatial = batch['dem_spatial'].to(device)
                    interaction_spatial = batch['interaction_spatial'].to(device)
                    tigge_seq = batch['tigge_seq'].to(device)
                    time_features_t = batch['time_features_t'].to(device)
                    time_features_seq = batch['time_features_seq'].to(device)
                    target = batch['target'].to(device)

                    with autocast():
                        point_pred, mu, sigma, interval_pred = model(
                            tigge_spatial, dem_spatial, interaction_spatial,
                            tigge_seq, time_features_t, time_features_seq
                        )
                        loss, _ = criterion(point_pred, mu, sigma, interval_pred, target)

                    val_point_preds.append(point_pred.cpu().float())
                    val_mus.append(mu.cpu().float())
                    val_sigmas.append(sigma.cpu().float())
                    val_interval_preds.append(interval_pred.cpu().float())
                    val_targets.append(target.cpu().float())

                    val_loss += loss.item() * tigge_spatial.size(0)
                    val_batch_count += 1

                    if batch_idx % 100 == 0 or batch_idx == len(val_loader) - 1:
                        progress = (batch_idx + 1) / len(val_loader) * 100
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation progress: {progress:.1f}%")

                except Exception as e:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing validation batch {batch_idx}: {str(e)}")
                    continue

        val_loss = val_loss / len(val_loader.dataset) if val_batch_count > 0 else float('inf')
        val_losses.append(val_loss)

        # 计算验证指标
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing validation metrics")
        val_point_preds_tensor = torch.cat(val_point_preds)
        val_mus_tensor = torch.cat(val_mus)
        val_sigmas_tensor = torch.cat(val_sigmas)
        val_interval_preds_tensor = torch.cat(val_interval_preds)
        val_targets_tensor = torch.cat(val_targets)

        val_point_preds_flat = val_point_preds_tensor.numpy().flatten()
        val_mus_flat = val_mus_tensor.numpy().flatten()
        val_sigmas_flat = val_sigmas_tensor.numpy().flatten()
        val_targets_flat = val_targets_tensor.numpy().flatten()

        val_mae_val = np.mean(np.abs(val_point_preds_flat - val_targets_flat))
        val_rmse_val = np.sqrt(np.mean((val_point_preds_flat - val_targets_flat) ** 2))
        val_mean = np.mean(val_targets_flat)
        ss_tot = np.sum((val_targets_flat - val_mean) ** 2)
        ss_res = np.sum((val_targets_flat - val_point_preds_flat) ** 2)
        val_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        val_crps_val = compute_crps(
            torch.from_numpy(val_mus_flat),
            torch.from_numpy(val_sigmas_flat),
            torch.from_numpy(val_targets_flat)
        ).item()

        val_interval_metrics = compute_interval_metrics(
            val_interval_preds_tensor,
            val_targets_tensor,
            val_point_preds_tensor
        )

        val_mae.append(val_mae_val)
        val_rmse.append(val_rmse_val)
        val_r2.append(val_r2_val)
        val_crps.append(val_crps_val)
        val_cp_95.append(val_interval_metrics['CP_95'])
        val_cp_50.append(val_interval_metrics['CP_50'])
        val_mwp_95.append(val_interval_metrics['MWP_95'])
        val_mwp_50.append(val_interval_metrics['MWP_50'])
        val_mc_95.append(val_interval_metrics['MC_95'])
        val_mc_50.append(val_interval_metrics['MC_50'])

        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Validation Metrics:')
        print(f'  Loss: {val_loss:.6f} | MAE: {val_mae_val:.4f} | RMSE: {val_rmse_val:.4f} | R²: {val_r2_val:.4f}')
        print(f'  CRPS: {val_crps_val:.4f}')
        print(f'  CP_95: {val_interval_metrics["CP_95"]:.4f} | CP_50: {val_interval_metrics["CP_50"]:.4f}')
        print(f'  MWP_95: {val_interval_metrics["MWP_95"]:.4f} | MWP_50: {val_interval_metrics["MWP_50"]:.4f}')
        print(f'  MC_95: {val_interval_metrics["MC_95"]:.4f} | MC_50: {val_interval_metrics["MC_50"]:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints_transformer_task2/best_model_transformer_task2.pth')
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Model saved with val loss: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Early stopping at epoch {epoch + 1}')
                break

        # 保存检查点
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving checkpoint for epoch {epoch + 1}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_crps': train_crps,
            'val_crps': val_crps,
            'train_cp_95': train_cp_95,
            'train_cp_50': train_cp_50,
            'val_cp_95': val_cp_95,
            'val_cp_50': val_cp_50,
            'train_mwp_95': train_mwp_95,
            'train_mwp_50': train_mwp_50,
            'val_mwp_95': val_mwp_95,
            'val_mwp_50': val_mwp_50,
            'train_mc_95': train_mc_95,
            'train_mc_50': train_mc_50,
            'val_mc_95': val_mc_95,
            'val_mc_50': val_mc_50
        }, f'checkpoints_transformer_task2/checkpoint_epoch_{epoch + 1}_transformer_task2.pth')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} completed in {epoch_time / 60:.2f} minutes")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time remaining: {est_remaining_time / 3600:.2f} hours")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed in {total_time / 3600:.2f} hours")

    # 生成所有可视化图表
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating visualization plots")

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Transformer Baseline - Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 2. MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Transformer Baseline - Mean Absolute Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 3. RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Transformer Baseline - Root Mean Squared Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 4. R²曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_r2) + 1), train_r2, label='Train R²', color='blue', linewidth=2)
    plt.plot(range(1, len(val_r2) + 1), val_r2, label='Validation R²', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.title('Transformer Baseline - R² Score', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 5. CRPS曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_crps) + 1), train_crps, label='Train CRPS', color='blue', linewidth=2)
    plt.plot(range(1, len(val_crps) + 1), val_crps, label='Validation CRPS', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CRPS', fontsize=12)
    plt.title('Transformer Baseline - Continuous Ranked Probability Score', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('crps_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 6. 覆盖率曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_cp_95) + 1), train_cp_95, label='Train CP 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_cp_95) + 1), val_cp_95, label='Val CP 95%', color='orange', linewidth=2)
    axes[0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Coverage Probability', fontsize=12)
    axes[0].set_title('Transformer Baseline - 95% CI Coverage', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_cp_50) + 1), train_cp_50, label='Train CP 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_cp_50) + 1), val_cp_50, label='Val CP 50%', color='orange', linewidth=2)
    axes[1].axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='Target (50%)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Coverage Probability', fontsize=12)
    axes[1].set_title('Transformer Baseline - 50% CI Coverage', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('coverage_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 7. MWP曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_mwp_95) + 1), train_mwp_95, label='Train MWP 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_mwp_95) + 1), val_mwp_95, label='Val MWP 95%', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Width Percentage', fontsize=12)
    axes[0].set_title('Transformer Baseline - 95% Interval Width', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_mwp_50) + 1), train_mwp_50, label='Train MWP 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_mwp_50) + 1), val_mwp_50, label='Val MWP 50%', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Width Percentage', fontsize=12)
    axes[1].set_title('Transformer Baseline - 50% Interval Width', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('width_curve_transformer_task2.png', dpi=300)
    plt.close()

    # 8. MC曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_mc_95) + 1), train_mc_95, label='Train MC 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_mc_95) + 1), val_mc_95, label='Val MC 95%', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Coverage (Lower is Better)', fontsize=12)
    axes[0].set_title('Transformer Baseline - 95% Interval Efficiency', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_mc_50) + 1), train_mc_50, label='Train MC 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_mc_50) + 1), val_mc_50, label='Val MC 50%', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Coverage (Lower is Better)', fontsize=12)
    axes[1].set_title('Transformer Baseline - 50% Interval Efficiency', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('mc_curve_transformer_task2.png', dpi=300)
    plt.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All metric curves saved")

    # 生成预测可视化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction visualizations")
    visualize_predictions(model, val_loader, device, save_dir='visualizations_transformer_task2', num_samples=5)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training pipeline completed successfully!")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started (Transformer Baseline - Aligned with LTC)")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up training parameters")
    batch_size = 8
    epochs = 10
    learning_rate = 0.0002  # Transformer需要更小的学习率
    weight_decay = 0.00008  # Transformer适配的权重衰减
    patience = 3
    accumulation_steps = 4
    warmup_epochs = 2  # Warmup epochs

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading training dataset")
    train_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\train.nc", H, W)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading validation dataset")
    val_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\val.nc", H, W)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")

    # 初始化Transformer模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing Transformer model (Aligned with LTC)")
    model = TransformerWindSpeedPredictor(
        H, W,
        tigge_features=10,
        dropout_rate=0.15,  # Transformer适配的dropout（比RNN系列低）
        d_model=256,  # 模型维度
        nhead=8,  # 注意力头数（256/8=32，每个头32维）
        num_layers=3,  # Encoder层数
        dim_feedforward=512  # FFN中间维度
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model initialized successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Training")
    train_model(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        accumulation_steps=accumulation_steps,
        warmup_epochs=warmup_epochs
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed!")