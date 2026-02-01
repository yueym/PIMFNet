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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== MultiTaskHead（与LTC版本完全一致）====================
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim=216, dropout_rate=0.2):  # 修改1：input_dim改为216，与LTC一致
        super(MultiTaskHead, self).__init__()
        print(f"[INFO] MultiTaskHead initialized with input_dim={input_dim} (LSTM Baseline)")

        # 修改2：网络结构与LTC版本一致
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


# ==================== 修改：LSTM模型（与LTC架构对齐）====================
class LSTMWindSpeedPredictor(nn.Module):
    """
    LSTM风速预测模型 - 与LTC架构对齐版本
    修改要点：
    1. 删除空间CNN分支，只保留时序分支（与LTC一致）
    2. hidden_dim改为216（与LTC一致）
    3. 输出层结构与LTC一致
    """

    def __init__(self, H, W, tigge_features=10, dropout_rate=0.2,  # 修改3：dropout改为0.2与LTC一致
                 lstm_hidden_dim=216, lstm_layers=2):  # 修改4：hidden_dim改为216
        super(LSTMWindSpeedPredictor, self).__init__()
        print("[INFO] LSTMWindSpeedPredictor initialized - TEMPORAL-ONLY (Aligned with LTC architecture)")

        self.H = H
        self.W = W
        self.hidden_dim = lstm_hidden_dim

        # ==================== 修改5：删除空间CNN分支 ====================
        # 原代码：self.spatial_cnn = nn.Sequential(...)  # 已删除

        # ==================== 时序特征提取（LSTM） ====================
        # 输入维度与LTC一致：TIGGE(10) + 时间特征(5) = 15
        self.lstm = nn.LSTM(
            input_size=tigge_features + 5,  # 10 + 5 = 15，与LTC的ODEFunc输入一致
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # ==================== 修改6：输出层与LTC一致 ====================
        # LTC的output_layer结构
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, lstm_hidden_dim)
        )

        # ==================== 修改7：删除融合层 ====================
        # 原代码：self.fusion = nn.Sequential(...)  # 已删除

        # ==================== MultiTaskHead（input_dim=216与LTC一致） ====================
        self.multi_task_head = MultiTaskHead(
            input_dim=lstm_hidden_dim,  # 修改8：改为216
            dropout_rate=dropout_rate
        )

        # 权重初始化
        self._initialize_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] LSTM Model Total parameters: {total_params:,}")
        print(f"[INFO] LSTM Model Trainable parameters: {trainable_params:,}")

    def _initialize_weights(self):
        # LSTM特殊初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        # 输出层初始化
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):
        """
        修改9：forward流程与LTC对齐，只使用时序分支

        输入（保持与原Dataset兼容）：
            tigge_spatial: (B, 10, H, W) - 不使用
            dem_spatial: (B, 3, H, W) - 不使用
            interaction_spatial: (B, 5, H, W) - 不使用
            tigge_seq: (B, seq_len, 10, H, W) - 使用
            time_features_t: (B, 5) - 不使用
            time_features_seq: (B, seq_len, 5) - 使用
        """
        B, seq_len, C, H, W = tigge_seq.shape

        # ==================== 修改10：数据处理流程与LTC一致 ====================
        # LTC处理方式：将空间维度展平，每个空间点独立处理时序
        # tigge_seq: (B, seq_len, 10, H, W) -> (B*H*W, seq_len, 10)
        tigge_seq_reshaped = tigge_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, seq_len, C)

        # time_features_seq: (B, seq_len, 5) -> (B*H*W, seq_len, 5)
        time_seq_expanded = time_features_seq.unsqueeze(1).unsqueeze(2).expand(B, H, W, seq_len, 5)
        time_seq_expanded = time_seq_expanded.reshape(B * H * W, seq_len, 5)

        # 拼接TIGGE和时间特征：(B*H*W, seq_len, 15)
        lstm_input = torch.cat([tigge_seq_reshaped, time_seq_expanded], dim=-1)

        # ==================== LSTM处理时序 ====================
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # lstm_out: (B*H*W, seq_len, hidden_dim)

        # 取最后一个时间步的隐藏状态（与LTC取最后时刻一致）
        h = lstm_out[:, -1, :]  # (B*H*W, hidden_dim)

        # ==================== 修改11：通过输出层（与LTC一致） ====================
        h = torch.clamp(h, -100, 100)  # 与LTC一致的clamp
        out = self.output_layer(h)  # (B*H*W, hidden_dim)

        # 重塑回空间维度：(B*H*W, hidden_dim) -> (B, hidden_dim, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, hidden_dim, H, W)

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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] interaction_features range: [{interaction_min:.4f}, {interaction_max:.4f}]")

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


# ==================== CRPS计算（与LTC版本一致） ====================
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


# ==================== 区间指标计算（与LTC版本一致） ====================
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

    # 修改12：添加异常值检测（调试用）
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


# ==================== 多任务损失（与LTC版本一致） ====================
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.5, beta=0.25, gamma=0.26):  # 修改13：权重与LTC版本一致
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.point_criterion = nn.SmoothL1Loss()
        print(f"[INFO] MultiTaskLoss权重 (LSTM Baseline): α={alpha}, β={beta}, γ={gamma}")

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


# ==================== 可视化函数 ====================
def visualize_predictions(model, dataloader, device, save_dir='visualizations_LSTM_aligned', num_samples=5):
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
            plt.savefig(f'{save_dir}/prediction_sample_{batch_idx + 1}_LSTM_aligned.png', dpi=300, bbox_inches='tight')
            plt.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Visualizations saved to {save_dir}/")


# ==================== 训练函数（超参数与LTC版本一致） ====================
def train_model(model, train_loader, val_loader, device,
                epochs=10, learning_rate=0.00032, weight_decay=0.000045,  # 修改14：与LTC一致
                patience=3, accumulation_steps=4):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training process (LSTM Baseline - Aligned with LTC)")

    criterion = MultiTaskLoss(alpha=1.5, beta=0.25, gamma=0.26)  # 与LTC一致

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
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
    os.makedirs('checkpoints_LSTM_task2', exist_ok=True)  # 修改15：保存路径
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint directory created: checkpoints_LSTM_aligned")

    first_batch = next(iter(train_loader))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_seq shape: {first_batch['tigge_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_seq shape: {first_batch['time_features_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs}")
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

                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                    if batch_idx % 100 == 0:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gradient norm: {grad_norm:.4f} (before clip: {total_norm:.4f})")

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
        print(f'  Loss: {train_loss:.6f} | MAE: {train_mae_val:.4f} | RMSE: {train_rmse_val:.4f} | R²: {train_r2_val:.4f}')
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
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing validation batch {batch_idx}: {str(e)}")
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

        scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints_LSTM_task2/best_model_LSTM_task2.pth')
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
            'scheduler_state_dict': scheduler.state_dict(),
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
        }, f'checkpoints_LSTM_task2/checkpoint_epoch_{epoch + 1}_LSTM_task2.pth')

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
    plt.title('LSTM Baseline (Aligned) - Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 2. MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('LSTM Baseline (Aligned) - Mean Absolute Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 3. RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('LSTM Baseline (Aligned) - Root Mean Squared Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 4. R²曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_r2) + 1), train_r2, label='Train R²', color='blue', linewidth=2)
    plt.plot(range(1, len(val_r2) + 1), val_r2, label='Validation R²', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.title('LSTM Baseline (Aligned) - R² Score', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 5. CRPS曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_crps) + 1), train_crps, label='Train CRPS', color='blue', linewidth=2)
    plt.plot(range(1, len(val_crps) + 1), val_crps, label='Validation CRPS', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CRPS', fontsize=12)
    plt.title('LSTM Baseline (Aligned) - Continuous Ranked Probability Score', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('crps_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 6. 覆盖率曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_cp_95) + 1), train_cp_95, label='Train CP 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_cp_95) + 1), val_cp_95, label='Val CP 95%', color='orange', linewidth=2)
    axes[0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Coverage Probability', fontsize=12)
    axes[0].set_title('LSTM Baseline (Aligned) - 95% CI Coverage', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_cp_50) + 1), train_cp_50, label='Train CP 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_cp_50) + 1), val_cp_50, label='Val CP 50%', color='orange', linewidth=2)
    axes[1].axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='Target (50%)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Coverage Probability', fontsize=12)
    axes[1].set_title('LSTM Baseline (Aligned) - 50% CI Coverage', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('coverage_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 7. MWP曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_mwp_95) + 1), train_mwp_95, label='Train MWP 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_mwp_95) + 1), val_mwp_95, label='Val MWP 95%', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Width Percentage', fontsize=12)
    axes[0].set_title('LSTM Baseline (Aligned) - 95% Interval Width', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_mwp_50) + 1), train_mwp_50, label='Train MWP 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_mwp_50) + 1), val_mwp_50, label='Val MWP 50%', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Width Percentage', fontsize=12)
    axes[1].set_title('LSTM Baseline (Aligned) - 50% Interval Width', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('width_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    # 8. MC曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(train_mc_95) + 1), train_mc_95, label='Train MC 95%', color='blue', linewidth=2)
    axes[0].plot(range(1, len(val_mc_95) + 1), val_mc_95, label='Val MC 95%', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Coverage (Lower is Better)', fontsize=12)
    axes[0].set_title('LSTM Baseline (Aligned) - 95% Interval Efficiency', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, len(train_mc_50) + 1), train_mc_50, label='Train MC 50%', color='blue', linewidth=2)
    axes[1].plot(range(1, len(val_mc_50) + 1), val_mc_50, label='Val MC 50%', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Mean Coverage (Lower is Better)', fontsize=12)
    axes[1].set_title('LSTM Baseline (Aligned) - 50% Interval Efficiency', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('mc_curve_LSTM_aligned.png', dpi=300)
    plt.close()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All metric curves saved")

    # 生成预测可视化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction visualizations")
    visualize_predictions(model, val_loader, device, save_dir='visualizations_LSTM_aligned', num_samples=5)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training pipeline completed successfully!")


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started (LSTM Baseline - Aligned with LTC)")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up training parameters")
    batch_size = 8
    epochs = 10
    learning_rate = 0.00032  # 修改16：与LTC一致
    weight_decay = 0.000045  # 修改17：与LTC一致
    patience = 3
    accumulation_steps = 4

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading training dataset")
    train_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\train.nc", H, W)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading validation dataset")
    val_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\val.nc", H, W)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")

    # 修改18：初始化对齐后的LSTM模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing LSTM model (Aligned with LTC)")
    model = LSTMWindSpeedPredictor(
        H, W,
        tigge_features=10,
        dropout_rate=0.2,  # 与LTC一致
        lstm_hidden_dim=216,  # 与LTC一致
        lstm_layers=2  # 减少层数以匹配LTC复杂度
    ).to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model initialized successfully")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Training")
    train_model(
        model, train_loader, val_loader, device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        accumulation_steps=accumulation_steps
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed!")