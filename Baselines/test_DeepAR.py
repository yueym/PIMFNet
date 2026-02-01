import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import os
import time
import json
import gc
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# ==================== MultiTaskHead ====================
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim=128, dropout_rate=0.1):
        super(MultiTaskHead, self).__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(input_dim, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.point_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.prob_mu = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.prob_log_sigma = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.interval_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Softplus()
        )

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


# ==================== DeepAR核心组件 ====================
class GaussianOutput(nn.Module):
    """高斯分布输出层"""

    def __init__(self, input_dim, output_dim=1):
        super(GaussianOutput, self).__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.sigma_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = torch.sigmoid(self.mu_layer(x))
        log_sigma = self.sigma_layer(x)
        log_sigma = torch.clamp(log_sigma, min=-4.6, max=-1.2)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class DeepARCell(nn.Module):
    """DeepAR单步单元"""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(DeepARCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.gaussian_output = GaussianOutput(hidden_dim)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        mu, sigma = self.gaussian_output(lstm_out)
        return lstm_out, hidden, mu.squeeze(-1), sigma.squeeze(-1)


class DeepAREncoder(nn.Module):
    """DeepAR编码器"""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(DeepAREncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.deepar_cell = DeepARCell(hidden_dim, hidden_dim, num_layers, dropout)

        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.input_projection(x)
        lstm_out, hidden, mu_seq, sigma_seq = self.deepar_cell(x)
        context = lstm_out[:, -1, :]
        context = self.context_projection(context)
        mu_final = mu_seq[:, -1]
        sigma_final = sigma_seq[:, -1]
        return context, hidden, mu_final, sigma_final


# ==================== DeepAR风速预测模型 ====================
class DeepARWindSpeedPredictor(nn.Module):
    """DeepAR风速预测模型"""

    def __init__(self, H, W, tigge_features=10, dropout_rate=0.1,
                 hidden_dim=64, num_layers=1, output_dim=128):
        super(DeepARWindSpeedPredictor, self).__init__()

        self.H = H
        self.W = W
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        input_dim = tigge_features + 5

        self.deepar_encoder = DeepAREncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim // 2, output_dim)
        )

        self.multi_task_head = MultiTaskHead(
            input_dim=output_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):
        B, seq_len, C, H, W = tigge_seq.shape

        tigge_seq_reshaped = tigge_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, seq_len, C)
        time_seq_expanded = time_features_seq.unsqueeze(1).unsqueeze(2).expand(B, H, W, seq_len, 5)
        time_seq_expanded = time_seq_expanded.reshape(B * H * W, seq_len, 5)

        deepar_input = torch.cat([tigge_seq_reshaped, time_seq_expanded], dim=-1)

        context, hidden, deepar_mu, deepar_sigma = self.deepar_encoder(deepar_input)
        x = self.feature_fusion(context)
        x = torch.clamp(x, -50, 50)
        out = self.output_layer(x)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        point_pred, mu, sigma, interval_pred = self.multi_task_head(out)

        deepar_mu = deepar_mu.reshape(B, H, W)
        deepar_sigma = deepar_sigma.reshape(B, H, W)

        final_mu = 0.6 * mu + 0.4 * deepar_mu
        final_sigma = 0.6 * sigma + 0.4 * deepar_sigma

        return point_pred, final_mu, final_sigma, interval_pred


# ==================== 数据集类 ====================
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.samples_per_time = H * W

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")

        ds = xr.open_dataset(ds_path, cache=False)

        tigge_raw = ds['tigge_features'].values
        dem_raw = ds['dem_features'].values
        interaction_raw = ds['interaction_features'].values
        target_raw = ds['target'].values
        time_data = ds['time_features'].values

        total_samples = tigge_raw.shape[0]
        self.T = total_samples // self.samples_per_time

        self.tigge_features = tigge_raw.reshape(self.T, H, W, -1)
        self.dem_features = dem_raw.reshape(self.T, H, W, -1)
        self.interaction_features = interaction_raw.reshape(self.T, H, W, -1)
        self.target = target_raw.reshape(self.T, H, W)

        time_data_per_t = time_data.reshape(self.T, self.samples_per_time, -1)[:, 0, :]
        self.time_scaler = StandardScaler()
        self.time_features_normalized = self.time_scaler.fit_transform(time_data_per_t)

        # 保存原始时间特征用于重构日期
        self.time_features_original = time_data_per_t

        self.sample_indices = np.arange(self.T - self.seq_len + 1)

        ds.close()
        del ds, tigge_raw, dem_raw, interaction_raw, target_raw, time_data
        gc.collect()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset initialized with {self.T} time points")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Valid samples: {len(self.sample_indices)}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        t_start = self.sample_indices[idx]
        t_end = t_start + self.seq_len

        tigge_seq = self.tigge_features[t_start:t_end]
        time_features_seq = self.time_features_normalized[t_start:t_end]

        t_current = t_end - 1
        tigge_spatial = self.tigge_features[t_current]
        dem_spatial = self.dem_features[t_current]
        interaction_spatial = self.interaction_features[t_current]
        target = self.target[t_current]
        time_features_t = self.time_features_normalized[t_current]

        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),
            'interaction_spatial': torch.from_numpy(interaction_spatial).float().permute(2, 0, 1),
            'tigge_seq': torch.from_numpy(tigge_seq).float().permute(0, 3, 1, 2),
            'time_features_t': torch.from_numpy(time_features_t).float(),
            'time_features_seq': torch.from_numpy(time_features_seq).float(),
            'target': torch.from_numpy(target).float()
        }


# ==================== 指标计算函数 ====================
def calculate_point_metrics(pred, target):
    """计算点预测指标（8个）"""
    pred = pred.flatten()
    target = target.flatten()

    FA = np.mean(np.abs(pred - target) < 1) * 100
    RMSE = np.sqrt(np.mean((pred - target) ** 2))
    MAE = np.mean(np.abs(pred - target))
    mean_target = np.mean(target)
    rRMSE = (RMSE / mean_target) * 100 if mean_target > 0 else 0
    rMAE = (MAE / mean_target) * 100 if mean_target > 0 else 0

    if np.std(pred) > 0 and np.std(target) > 0:
        R = np.corrcoef(pred, target)[0, 1]
    else:
        R = 0

    ss_tot = np.sum((target - mean_target) ** 2)
    ss_res = np.sum((target - pred) ** 2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    epsilon = 1e-8
    target_safe = np.where(np.abs(target) < epsilon, epsilon, target)
    MAPE = np.mean(np.abs((pred - target) / target_safe)) * 100

    return {
        'FA': float(FA),
        'RMSE': float(RMSE),
        'MAE': float(MAE),
        'rRMSE': float(rRMSE),
        'rMAE': float(rMAE),
        'R': float(R),
        'R2': float(R2),
        'MAPE': float(MAPE)
    }


def calculate_interval_metrics(q_0025, q_025, q_075, q_0975, target, point_pred):
    """计算区间预测指标（8个）"""
    coverage_95 = np.mean((target >= q_0025) & (target <= q_0975))
    coverage_50 = np.mean((target >= q_025) & (target <= q_075))

    width_95 = q_0975 - q_0025
    width_50 = q_075 - q_025

    point_pred_safe = np.clip(np.abs(point_pred), 1e-3, None)
    mwp_95 = np.mean(width_95 / point_pred_safe)
    mwp_50 = np.mean(width_50 / point_pred_safe)

    mc_95 = mwp_95 / max(coverage_95, 1e-6)
    mc_50 = mwp_50 / max(coverage_50, 1e-6)

    target_range = np.max(target) - np.min(target) + 1e-6
    pinaw_95 = np.mean(width_95) / target_range
    pinaw_50 = np.mean(width_50) / target_range

    return {
        'CP_95': float(coverage_95),
        'CP_50': float(coverage_50),
        'MWP_95': float(mwp_95),
        'MWP_50': float(mwp_50),
        'MC_95': float(mc_95),
        'MC_50': float(mc_50),
        'PINAW_95': float(pinaw_95),
        'PINAW_50': float(pinaw_50)
    }


def calculate_crps(mu, sigma, target):
    """计算CRPS"""
    sigma = np.clip(sigma, 0.01, 50)
    z = (target - mu) / sigma
    z = np.clip(z, -5.0, 5.0)

    phi_z = stats.norm.cdf(z)
    pdf_z = stats.norm.pdf(z)

    crps = sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1.0 / np.sqrt(np.pi))
    crps_mean = np.mean(np.abs(crps))

    return float(crps_mean)


# ==================== 测试函数 ====================
def test_deepar_model(test_path, model_path, save_dir, H=48, W=96, seq_len=4):
    """测试DeepAR模型并保存所有指标和预测结果"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting DeepAR model testing...")

    os.makedirs(save_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # ========== 1. 加载测试数据 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading test dataset...")
    test_dataset = WindDataset(test_path, H, W, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test dataset size: {len(test_dataset)}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Number of test batches: {len(test_loader)}")

    # ========== 2. 加载模型 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading DeepAR model...")

    model = DeepARWindSpeedPredictor(
        H=H, W=W,
        tigge_features=10,
        dropout_rate=0.1,
        hidden_dim=64,
        num_layers=1,
        output_dim=128
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully!")

    # ========== 3. 生成预测 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Making predictions...")

    all_point_preds = []
    all_mus = []
    all_sigmas = []
    all_intervals = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {batch_idx + 1}/{len(test_loader)}")

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

            all_point_preds.append(point_pred.cpu().numpy())
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_intervals.append(interval_pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # 拼接所有批次
    point_pred_3d = np.concatenate(all_point_preds, axis=0)
    mu_3d = np.concatenate(all_mus, axis=0)
    sigma_3d = np.concatenate(all_sigmas, axis=0)
    interval_preds = np.concatenate(all_intervals, axis=0)
    target_3d = np.concatenate(all_targets, axis=0)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Predictions generated:")
    print(f"  Point predictions shape: {point_pred_3d.shape}")
    print(f"  Intervals shape: {interval_preds.shape}")
    print(f"  Targets shape: {target_3d.shape}")

    # ========== 4. 加载标准化器并反标准化 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading scalers...")
    scaler_target = joblib.load(r'E:\yym2\qixiang\Obs_PDF\final\target_scaler.pkl')

    # 获取反标准化参数
    if hasattr(scaler_target, 'data_min_'):
        target_data_min = scaler_target.data_min_[0]
        target_range = 1 / scaler_target.scale_[0]
        target_scaler_type = 'MinMaxScaler'
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: MinMaxScaler")
    elif hasattr(scaler_target, 'center_'):
        target_center = scaler_target.center_[0]
        target_scale = scaler_target.scale_[0]
        target_scaler_type = 'RobustScaler'
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: RobustScaler")
    elif hasattr(scaler_target, 'mean_'):
        target_mean = scaler_target.mean_[0]
        target_std = scaler_target.scale_[0]
        target_scaler_type = 'StandardScaler'
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: StandardScaler")

    # 定义反标准化函数
    def inverse_transform_target(data):
        if target_scaler_type == 'MinMaxScaler':
            return (data * target_range) + target_data_min
        elif target_scaler_type == 'RobustScaler':
            return (data * target_scale) + target_center
        elif target_scaler_type == 'StandardScaler':
            return (data * target_std) + target_mean

    def inverse_transform_sigma(data):
        if target_scaler_type == 'MinMaxScaler':
            return data * target_range
        elif target_scaler_type == 'RobustScaler':
            return data * target_scale
        elif target_scaler_type == 'StandardScaler':
            return data * target_std

    # 反标准化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inverse transforming predictions...")
    point_pred_orig = inverse_transform_target(point_pred_3d)
    target_orig = inverse_transform_target(target_3d)
    mu_orig = inverse_transform_target(mu_3d)
    sigma_orig = inverse_transform_sigma(sigma_3d)

    # 区间预测反标准化
    q_0025_orig = inverse_transform_target(interval_preds[:, 0, :, :])
    q_025_orig = inverse_transform_target(interval_preds[:, 1, :, :])
    q_075_orig = inverse_transform_target(interval_preds[:, 2, :, :])
    q_0975_orig = inverse_transform_target(interval_preds[:, 3, :, :])

    # 裁剪到合理范围
    point_pred_orig = np.clip(point_pred_orig, 0, 100)
    target_orig = np.clip(target_orig, 0, 100)
    mu_orig = np.clip(mu_orig, 0, 100)
    sigma_orig = np.clip(sigma_orig, 0.01, 50)
    q_0025_orig = np.clip(q_0025_orig, 0, 100)
    q_025_orig = np.clip(q_025_orig, 0, 100)
    q_075_orig = np.clip(q_075_orig, 0, 100)
    q_0975_orig = np.clip(q_0975_orig, 0, 100)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] After inverse transform:")
    print(f"  Point predictions range: [{point_pred_orig.min():.4f}, {point_pred_orig.max():.4f}]")
    print(f"  Targets range: [{target_orig.min():.4f}, {target_orig.max():.4f}]")

    # ========== 5. 重构日期信息 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reconstructing dates...")

    # 从数据集获取时间特征
    time_features = test_dataset.time_features_original[seq_len - 1:]
    dates = pd.to_datetime({
        'year': time_features[:, 0].astype(int),
        'month': time_features[:, 1].astype(int),
        'day': time_features[:, 2].astype(int),
        'hour': time_features[:, 3].astype(int)
    })

    # 修改：确保转换为DatetimeIndex，并保存为numpy数组用于保存
    if isinstance(dates, pd.Series):
        dates_pd = pd.DatetimeIndex(dates)
        dates_array = dates.values  # 用于保存的numpy数组
    else:
        dates_pd = dates
        dates_array = dates

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Reconstructed {len(dates_pd)} dates")

    # ========== 6. 保存原始预测结果 ==========
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving prediction results...")
    np.save(os.path.join(save_dir, 'DeepAR_test_point_preds.npy'), point_pred_orig)
    np.save(os.path.join(save_dir, 'DeepAR_test_targets.npy'), target_orig)
    np.save(os.path.join(save_dir, 'DeepAR_test_dates.npy'), dates_array)  # 使用dates_array
    np.save(os.path.join(save_dir, 'DeepAR_test_mus.npy'), mu_orig)
    np.save(os.path.join(save_dir, 'DeepAR_test_sigmas.npy'), sigma_orig)

    # 保存区间预测 - 格式: (T, 4, H, W)
    interval_preds_orig = np.stack([q_0025_orig, q_025_orig, q_075_orig, q_0975_orig], axis=1)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Interval predictions shape: {interval_preds_orig.shape}")
    np.save(os.path.join(save_dir, 'DeepAR_test_intervals.npy'), interval_preds_orig)

    # 打印保存文件的详细信息
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved files summary:")
    print(f"  ✓ DeepAR_test_point_preds.npy : shape {point_pred_orig.shape}")
    print(f"  ✓ DeepAR_test_targets.npy : shape {target_orig.shape}")
    print(f"  ✓ DeepAR_test_intervals.npy : shape {interval_preds_orig.shape} (4 quantiles)")
    print(f"  ✓ DeepAR_test_mus.npy : shape {mu_orig.shape}")
    print(f"  ✓ DeepAR_test_sigmas.npy : shape {sigma_orig.shape}")
    print(f"  ✓ DeepAR_test_dates.npy : shape {dates_array.shape}")  # 修改
    print(f"\n  Interval quantiles order: [q_0.025, q_0.25, q_0.75, q_0.975]")

    # ========== 7. 辅助函数 ==========
    def to_numpy_mask(mask):
        if hasattr(mask, 'values'):
            return mask.values
        return np.array(mask)

    def compute_all_metrics_for_subset(mask, subset_name):
        mask = to_numpy_mask(mask)
        if np.sum(mask) == 0:
            print(f"[WARNING] No samples for {subset_name}")
            return None

        subset_preds = point_pred_orig[mask]
        subset_targets = target_orig[mask]
        subset_mus = mu_orig[mask]
        subset_sigmas = sigma_orig[mask]
        subset_q_0025 = q_0025_orig[mask]
        subset_q_025 = q_025_orig[mask]
        subset_q_075 = q_075_orig[mask]
        subset_q_0975 = q_0975_orig[mask]

        point_metrics = calculate_point_metrics(subset_preds, subset_targets)
        interval_metrics = calculate_interval_metrics(
            subset_q_0025.flatten(), subset_q_025.flatten(),
            subset_q_075.flatten(), subset_q_0975.flatten(),
            subset_targets.flatten(), subset_preds.flatten()
        )
        crps = calculate_crps(subset_mus.flatten(), subset_sigmas.flatten(), subset_targets.flatten())

        return {
            'point': point_metrics,
            'interval': interval_metrics,
            'probabilistic': {'CRPS': crps}
        }

    def print_full_metrics(metrics, period_name):
        print(f"\n{'=' * 70}")
        print(f"[{period_name}] COMPLETE METRICS")
        print(f"{'=' * 70}")

        print(f"\n[{period_name}] Point Prediction (8 metrics):")
        print(f"  {'Metric':<12} {'Value':>12}")
        print(f"  {'-' * 24}")
        for metric in ['FA', 'RMSE', 'MAE', 'rRMSE', 'rMAE', 'R', 'R2', 'MAPE']:
            if metric in metrics['point']:
                print(f"  {metric:<12} {metrics['point'][metric]:>12.4f}")

        print(f"\n[{period_name}] Interval Prediction (8 metrics):")
        print(f"  {'Metric':<12} {'95% CI':>12} {'50% CI':>12}")
        print(f"  {'-' * 36}")
        print(f"  {'CP':<12} {metrics['interval']['CP_95']:>12.4f} {metrics['interval']['CP_50']:>12.4f}")
        print(f"  {'MWP':<12} {metrics['interval']['MWP_95']:>12.4f} {metrics['interval']['MWP_50']:>12.4f}")
        print(f"  {'MC':<12} {metrics['interval']['MC_95']:>12.4f} {metrics['interval']['MC_50']:>12.4f}")
        print(f"  {'PINAW':<12} {metrics['interval']['PINAW_95']:>12.4f} {metrics['interval']['PINAW_50']:>12.4f}")

        print(f"\n[{period_name}] Probabilistic Prediction (1 metric):")
        print(f"  CRPS: {metrics['probabilistic']['CRPS']:.4f}")

    # 使用dates_pd（DatetimeIndex）进行时间筛选
    T_test = len(dates_pd)

    # ========== 8. 年度指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== YEARLY METRICS =====")
    yearly_mask = np.ones(T_test, dtype=bool)
    yearly_mask_expanded = np.tile(yearly_mask.reshape(-1, 1, 1), (1, H, W))
    yearly_metrics = compute_all_metrics_for_subset(yearly_mask_expanded, "Yearly")
    print_full_metrics(yearly_metrics, "YEARLY")

    with open(os.path.join(save_dir, 'DeepAR_yearly_metrics_point.json'), 'w') as f:
        json.dump(yearly_metrics['point'], f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_yearly_metrics_interval.json'), 'w') as f:
        json.dump(yearly_metrics['interval'], f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_yearly_metrics_probabilistic.json'), 'w') as f:
        json.dump(yearly_metrics['probabilistic'], f, indent=2)

    # ========== 9. 季节指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== SEASONAL METRICS =====")
    seasons = {
        'Spring': (3, 5),
        'Summer': (6, 8),
        'Autumn': (9, 11),
        'Winter': (12, 2)
    }
    seasonal_metrics_point = {}
    seasonal_metrics_interval = {}
    seasonal_metrics_probabilistic = {}

    for season, (start_month, end_month) in seasons.items():
        if start_month < end_month:
            mask = (dates_pd.month >= start_month) & (dates_pd.month <= end_month)
        else:
            mask = (dates_pd.month >= start_month) | (dates_pd.month <= end_month)

        mask_np = to_numpy_mask(mask)
        mask_expanded = np.tile(mask_np.reshape(-1, 1, 1), (1, H, W))
        metrics = compute_all_metrics_for_subset(mask_expanded, season)
        if metrics:
            seasonal_metrics_point[season] = metrics['point']
            seasonal_metrics_interval[season] = metrics['interval']
            seasonal_metrics_probabilistic[season] = metrics['probabilistic']
            print_full_metrics(metrics, season.upper())

    with open(os.path.join(save_dir, 'DeepAR_seasonal_metrics_point.json'), 'w') as f:
        json.dump(seasonal_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_seasonal_metrics_interval.json'), 'w') as f:
        json.dump(seasonal_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_seasonal_metrics_probabilistic.json'), 'w') as f:
        json.dump(seasonal_metrics_probabilistic, f, indent=2)

    # ========== 10. 月份指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MONTHLY METRICS =====")
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_metrics_point = {}
    monthly_metrics_interval = {}
    monthly_metrics_probabilistic = {}

    for month in range(1, 13):
        mask = (dates_pd.month == month)
        mask_np = to_numpy_mask(mask)
        mask_expanded = np.tile(mask_np.reshape(-1, 1, 1), (1, H, W))
        metrics = compute_all_metrics_for_subset(mask_expanded, months_names[month - 1])
        if metrics:
            monthly_metrics_point[str(month)] = metrics['point']
            monthly_metrics_interval[str(month)] = metrics['interval']
            monthly_metrics_probabilistic[str(month)] = metrics['probabilistic']
            print_full_metrics(metrics, months_names[month - 1].upper())

    with open(os.path.join(save_dir, 'DeepAR_monthly_metrics_point.json'), 'w') as f:
        json.dump(monthly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_monthly_metrics_interval.json'), 'w') as f:
        json.dump(monthly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_monthly_metrics_probabilistic.json'), 'w') as f:
        json.dump(monthly_metrics_probabilistic, f, indent=2)

    # ========== 11. 四个时间点指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== HOURLY METRICS (4 time points) =====")
    hours = [0, 6, 12, 18]
    hourly_metrics_point = {}
    hourly_metrics_interval = {}
    hourly_metrics_probabilistic = {}

    for hour in hours:
        mask = (dates_pd.hour == hour)
        mask_np = to_numpy_mask(mask)
        mask_expanded = np.tile(mask_np.reshape(-1, 1, 1), (1, H, W))
        hour_str = f"{hour:02d}:00"
        metrics = compute_all_metrics_for_subset(mask_expanded, hour_str)
        if metrics:
            hourly_metrics_point[str(hour)] = metrics['point']
            hourly_metrics_interval[str(hour)] = metrics['interval']
            hourly_metrics_probabilistic[str(hour)] = metrics['probabilistic']
            print_full_metrics(metrics, hour_str)

    with open(os.path.join(save_dir, 'DeepAR_hourly_metrics_point.json'), 'w') as f:
        json.dump(hourly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_hourly_metrics_interval.json'), 'w') as f:
        json.dump(hourly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'DeepAR_hourly_metrics_probabilistic.json'), 'w') as f:
        json.dump(hourly_metrics_probabilistic, f, indent=2)

    # ========== 12. 保存区间可视化数据 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving interval predictions for visualization...")
    np.savez(
        os.path.join(save_dir, 'DeepAR_interval_visualization_data.npz'),
        dates=dates_array,  # 使用dates_array
        point_pred=point_pred_orig,
        target=target_orig,
        q_0025=q_0025_orig,
        q_025=q_025_orig,
        q_075=q_075_orig,
        q_0975=q_0975_orig
    )


    # 最终汇总
    print(f"\n\n{'#' * 80}")
    print(f"{'FINAL SUMMARY - DeepAR MODEL':^80}")
    print(f"{'#' * 80}")
    print(f"\n[Files Saved to: {save_dir}]")
    print(f"\nSaved Files:")
    print(f"  - DeepAR_yearly_metrics_point.json")
    print(f"  - DeepAR_yearly_metrics_interval.json")
    print(f"  - DeepAR_yearly_metrics_probabilistic.json")
    print(f"  - DeepAR_seasonal_metrics_*.json")
    print(f"  - DeepAR_monthly_metrics_*.json")
    print(f"  - DeepAR_hourly_metrics_*.json")
    print(f"  - DeepAR_test_*.npy")
    print(f"  - DeepAR_interval_visualization_data.npz")
    print(f"\n[Interval Predictions]:")
    print(f"  - DeepAR_test_intervals.npy : shape {interval_preds_orig.shape}")
    print(f"    Format: (Time, 4_quantiles, Height, Width)")
    print(f"    Quantiles: [q_0.025, q_0.25, q_0.75, q_0.975]")
    print(f"    Represents: [95%_lower, 50%_lower, 50%_upper, 95%_upper]")

    print(f"\n{'=' * 80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed successfully!")
    print(f"{'=' * 80}")

    return {
        'yearly': yearly_metrics,
        'seasonal': {
            'point': seasonal_metrics_point,
            'interval': seasonal_metrics_interval,
            'probabilistic': seasonal_metrics_probabilistic
        },
        'monthly': {
            'point': monthly_metrics_point,
            'interval': monthly_metrics_interval,
            'probabilistic': monthly_metrics_probabilistic
        },
        'hourly': {
            'point': hourly_metrics_point,
            'interval': hourly_metrics_interval,
            'probabilistic': hourly_metrics_probabilistic
        }
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    import joblib

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== DeepAR MODEL TESTING =====")

    # 设置参数
    H, W = 48, 96
    seq_len = 4

    # 路径设置
    test_path = r"E:\yym2\qixiang\Obs_PDF\final\test.nc"
    model_path = r"E:\yym2\qixiang\Obs_PDF\src\checkpoints_DeepAR_task2\best_model_DeepAR_task2.pth"
    save_dir = r"E:\yym2\qixiang\Obs_PDF\src\resluts_task2\DeepAR"

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Configuration:")
    print(f"  Test data path: {test_path}")
    print(f"  Model path: {model_path}")
    print(f"  Save directory: {save_dir}")
    print(f"  Spatial dimensions: {H} x {W}")
    print(f"  Sequence length: {seq_len}")

    # 运行测试
    results = test_deepar_model(test_path, model_path, save_dir, H, W, seq_len)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== ALL TESTING COMPLETED =====")

