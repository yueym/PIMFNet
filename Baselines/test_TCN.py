import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import os
import time
import json
import joblib
from torch.cuda.amp import autocast
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F


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


# ==================== TCN核心组件 ====================
class Chomp1d(nn.Module):
    """裁剪模块：移除因果卷积产生的多余padding"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """TCN基本块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN主干网络"""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==================== TCN风速预测模型 ====================
class TCNWindSpeedPredictor(nn.Module):
    """TCN风速预测模型 - 多任务版本"""

    def __init__(self, H, W, tigge_features=10, dropout_rate=0.1,
                 tcn_channels=[32, 64, 96], kernel_size=3, output_dim=128):
        super(TCNWindSpeedPredictor, self).__init__()

        self.H = H
        self.W = W
        self.output_dim = output_dim

        input_dim = tigge_features + 5  # 15

        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, tcn_channels[0]),
            nn.LayerNorm(tcn_channels[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # TCN主干
        self.tcn = TemporalConvNet(
            num_inputs=tcn_channels[0],
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout_rate
        )

        # 时序注意力池化
        self.temporal_attention = nn.Sequential(
            nn.Linear(tcn_channels[-1], tcn_channels[-1] // 2),
            nn.Tanh(),
            nn.Linear(tcn_channels[-1] // 2, 1)
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(tcn_channels[-1], output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(output_dim // 2, output_dim)
        )

        # MultiTaskHead
        self.multi_task_head = MultiTaskHead(
            input_dim=output_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):
        B, seq_len, C, H, W = tigge_seq.shape

        # 重塑为 (B*H*W, seq_len, C)
        tigge_seq_reshaped = tigge_seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, seq_len, C)

        # 扩展时间特征
        time_seq_expanded = time_features_seq.unsqueeze(1).unsqueeze(2).expand(B, H, W, seq_len, 5)
        time_seq_expanded = time_seq_expanded.reshape(B * H * W, seq_len, 5)

        # 拼接特征
        tcn_input = torch.cat([tigge_seq_reshaped, time_seq_expanded], dim=-1)

        # 输入投影
        x = self.input_projection(tcn_input)

        # TCN处理 (需要 channels-first)
        x = x.permute(0, 2, 1)  # (B*H*W, channels, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)  # (B*H*W, seq_len, channels)

        # 注意力池化
        attn_weights = self.temporal_attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)  # (B*H*W, channels)

        # 特征融合
        x = self.feature_fusion(x)

        # 输出层
        x = torch.clamp(x, -50, 50)
        out = self.output_layer(x)

        # 重塑为空间格式
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # MultiTaskHead
        point_pred, mu, sigma, interval_pred = self.multi_task_head(out)

        return point_pred, mu, sigma, interval_pred


# ==================== WindDataset ====================
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        time_data = self.ds['time_features'].values
        self.time_scaler = StandardScaler()
        normalized_time = self.time_scaler.fit_transform(time_data)

        self.ds['time_features_normalized'] = xr.DataArray(
            normalized_time,
            dims=self.ds['time_features'].dims,
            coords={'sample': self.ds['time_features'].coords['sample']}
        )

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
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset initialized with {len(self.sample_indices)} samples")

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


def calculate_interval_metrics(interval_pred, target, point_pred):
    """计算区间预测指标（8个）"""
    q_0025 = interval_pred[:, 0, :, :].flatten()
    q_025 = interval_pred[:, 1, :, :].flatten()
    q_075 = interval_pred[:, 2, :, :].flatten()
    q_0975 = interval_pred[:, 3, :, :].flatten()
    target_flat = target.flatten()
    point_pred_flat = point_pred.flatten()

    coverage_95 = np.mean((target_flat >= q_0025) & (target_flat <= q_0975))
    coverage_50 = np.mean((target_flat >= q_025) & (target_flat <= q_075))

    width_95 = q_0975 - q_0025
    width_50 = q_075 - q_025

    point_pred_safe = np.clip(np.abs(point_pred_flat), 1e-3, None)
    mwp_95 = np.mean(width_95 / point_pred_safe)
    mwp_50 = np.mean(width_50 / point_pred_safe)

    mc_95 = mwp_95 / max(coverage_95, 1e-6)
    mc_50 = mwp_50 / max(coverage_50, 1e-6)

    target_range = np.max(target_flat) - np.min(target_flat) + 1e-6
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
    mu = mu.flatten()
    sigma = sigma.flatten()
    target = target.flatten()

    sigma = np.clip(sigma, 0.01, 50)
    z = (target - mu) / sigma
    z = np.clip(z, -5.0, 5.0)

    from scipy import stats
    phi_z = stats.norm.cdf(z)
    pdf_z = stats.norm.pdf(z)

    crps = sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1.0 / np.sqrt(np.pi))
    crps_mean = np.mean(np.abs(crps))

    return float(crps_mean)


# ==================== 测试函数 ====================
def test_model(model, test_loader, device, save_dir):
    """测试模型并保存所有指标和预测结果"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting model testing...")

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 加载标准化器
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

    # 收集所有预测结果
    all_point_preds = []
    all_mus = []
    all_sigmas = []
    all_interval_preds = []
    all_targets = []
    all_dates = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

                all_point_preds.append(point_pred.cpu().float().numpy())
                all_mus.append(mu.cpu().float().numpy())
                all_sigmas.append(sigma.cpu().float().numpy())
                all_interval_preds.append(interval_pred.cpu().float().numpy())
                all_targets.append(target.cpu().float().numpy())

                # 提取日期
                actual_indices = test_loader.dataset.sample_indices[
                                 batch_idx * test_loader.batch_size:
                                 (batch_idx + 1) * test_loader.batch_size
                                 ]
                batch_dates = [test_loader.dataset.time_points[idx + test_loader.dataset.seq_len - 1]
                               for idx in actual_indices
                               if idx + test_loader.dataset.seq_len - 1 < len(test_loader.dataset.time_points)]
                all_dates.extend(batch_dates)

                if batch_idx % 50 == 0 or batch_idx == len(test_loader) - 1:
                    progress = (batch_idx + 1) / len(test_loader) * 100
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test progress: {progress:.1f}%")

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error processing batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # 合并所有结果
    all_point_preds = np.concatenate(all_point_preds, axis=0)
    all_mus = np.concatenate(all_mus, axis=0)
    all_sigmas = np.concatenate(all_sigmas, axis=0)
    all_interval_preds = np.concatenate(all_interval_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_dates = np.array(all_dates[:len(all_point_preds)])

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total samples: {len(all_point_preds)}")

    # 反标准化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inverse transforming predictions...")
    all_point_preds_orig = inverse_transform_target(all_point_preds)
    all_targets_orig = inverse_transform_target(all_targets)
    all_mus_orig = inverse_transform_target(all_mus)
    all_sigmas_orig = inverse_transform_sigma(all_sigmas)
    all_interval_preds_orig = inverse_transform_target(all_interval_preds)

    # 裁剪到合理范围
    all_point_preds_orig = np.clip(all_point_preds_orig, 0, 100)
    all_targets_orig = np.clip(all_targets_orig, 0, 100)
    all_mus_orig = np.clip(all_mus_orig, 0, 100)
    all_sigmas_orig = np.clip(all_sigmas_orig, 0.01, 50)
    all_interval_preds_orig = np.clip(all_interval_preds_orig, 0, 100)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] After inverse transform:")
    print(f"  Point predictions range: [{all_point_preds_orig.min():.4f}, {all_point_preds_orig.max():.4f}]")
    print(f"  Targets range: [{all_targets_orig.min():.4f}, {all_targets_orig.max():.4f}]")
    print(f"  Mu range: [{all_mus_orig.min():.4f}, {all_mus_orig.max():.4f}]")
    print(f"  Sigma range: [{all_sigmas_orig.min():.4f}, {all_sigmas_orig.max():.4f}]")
    print(f"  Interval range: [{all_interval_preds_orig.min():.4f}, {all_interval_preds_orig.max():.4f}]")

    # 保存原始预测结果
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving prediction results...")
    np.save(os.path.join(save_dir, 'TCN_test_point_preds.npy'), all_point_preds_orig)
    np.save(os.path.join(save_dir, 'TCN_test_targets.npy'), all_targets_orig)
    np.save(os.path.join(save_dir, 'TCN_test_dates.npy'), all_dates)
    np.save(os.path.join(save_dir, 'TCN_test_mus.npy'), all_mus_orig)
    np.save(os.path.join(save_dir, 'TCN_test_sigmas.npy'), all_sigmas_orig)
    np.save(os.path.join(save_dir, 'TCN_test_intervals.npy'), all_interval_preds_orig)

    # 辅助函数
    def to_numpy_mask(mask):
        if hasattr(mask, 'values'):
            return mask.values
        return np.array(mask)

    def compute_all_metrics_for_subset(mask, subset_name):
        mask = to_numpy_mask(mask)
        if np.sum(mask) == 0:
            print(f"[WARNING] No samples for {subset_name}")
            return None

        subset_preds = all_point_preds_orig[mask]
        subset_targets = all_targets_orig[mask]
        subset_mus = all_mus_orig[mask]
        subset_sigmas = all_sigmas_orig[mask]
        subset_intervals = all_interval_preds_orig[mask]

        point_metrics = calculate_point_metrics(subset_preds, subset_targets)
        interval_metrics = calculate_interval_metrics(subset_intervals, subset_targets, subset_preds)
        crps = calculate_crps(subset_mus, subset_sigmas, subset_targets)

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

        # 转换日期格式

    dates_pd = pd.to_datetime(all_dates)

    # ========== 1. 年度指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== YEARLY METRICS =====")
    yearly_mask = np.ones(len(all_dates), dtype=bool)
    yearly_metrics = compute_all_metrics_for_subset(yearly_mask, "Yearly")
    print_full_metrics(yearly_metrics, "YEARLY")

    with open(os.path.join(save_dir, 'TCN_yearly_metrics_point.json'), 'w') as f:
        json.dump(yearly_metrics['point'], f, indent=2)
    with open(os.path.join(save_dir, 'TCN_yearly_metrics_interval.json'), 'w') as f:
        json.dump(yearly_metrics['interval'], f, indent=2)
    with open(os.path.join(save_dir, 'TCN_yearly_metrics_probabilistic.json'), 'w') as f:
        json.dump(yearly_metrics['probabilistic'], f, indent=2)

    # ========== 2. 季节指标 ==========
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
        metrics = compute_all_metrics_for_subset(mask_np, season)
        if metrics:
            seasonal_metrics_point[season] = metrics['point']
            seasonal_metrics_interval[season] = metrics['interval']
            seasonal_metrics_probabilistic[season] = metrics['probabilistic']
            print_full_metrics(metrics, season.upper())

    with open(os.path.join(save_dir, 'TCN_seasonal_metrics_point.json'), 'w') as f:
        json.dump(seasonal_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_seasonal_metrics_interval.json'), 'w') as f:
        json.dump(seasonal_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_seasonal_metrics_probabilistic.json'), 'w') as f:
        json.dump(seasonal_metrics_probabilistic, f, indent=2)

    # ========== 3. 月份指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MONTHLY METRICS =====")
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_metrics_point = {}
    monthly_metrics_interval = {}
    monthly_metrics_probabilistic = {}

    for month in range(1, 13):
        mask = (dates_pd.month == month)
        mask_np = to_numpy_mask(mask)
        metrics = compute_all_metrics_for_subset(mask_np, months_names[month - 1])
        if metrics:
            monthly_metrics_point[str(month)] = metrics['point']
            monthly_metrics_interval[str(month)] = metrics['interval']
            monthly_metrics_probabilistic[str(month)] = metrics['probabilistic']
            print_full_metrics(metrics, months_names[month - 1].upper())

    with open(os.path.join(save_dir, 'TCN_monthly_metrics_point.json'), 'w') as f:
        json.dump(monthly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_monthly_metrics_interval.json'), 'w') as f:
        json.dump(monthly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_monthly_metrics_probabilistic.json'), 'w') as f:
        json.dump(monthly_metrics_probabilistic, f, indent=2)

    # ========== 4. 四个时间点指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== HOURLY METRICS (4 time points) =====")
    hours = [0, 6, 12, 18]
    hourly_metrics_point = {}
    hourly_metrics_interval = {}
    hourly_metrics_probabilistic = {}

    for hour in hours:
        mask = (dates_pd.hour == hour)
        mask_np = to_numpy_mask(mask)
        hour_str = f"{hour:02d}:00"
        metrics = compute_all_metrics_for_subset(mask_np, hour_str)
        if metrics:
            hourly_metrics_point[str(hour)] = metrics['point']
            hourly_metrics_interval[str(hour)] = metrics['interval']
            hourly_metrics_probabilistic[str(hour)] = metrics['probabilistic']
            print_full_metrics(metrics, hour_str)

    with open(os.path.join(save_dir, 'TCN_hourly_metrics_point.json'), 'w') as f:
        json.dump(hourly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_hourly_metrics_interval.json'), 'w') as f:
        json.dump(hourly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'TCN_hourly_metrics_probabilistic.json'), 'w') as f:
        json.dump(hourly_metrics_probabilistic, f, indent=2)

    # ========== 5. 保存区间可视化数据 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving interval predictions for visualization...")
    np.savez(
        os.path.join(save_dir, 'TCN_interval_visualization_data.npz'),
        dates=all_dates,
        point_pred=all_point_preds_orig,
        target=all_targets_orig,
        q_0025=all_interval_preds_orig[:, 0, :, :],
        q_025=all_interval_preds_orig[:, 1, :, :],
        q_075=all_interval_preds_orig[:, 2, :, :],
        q_0975=all_interval_preds_orig[:, 3, :, :]
    )

    # 最终汇总
    print(f"\n\n{'#' * 80}")
    print(f"{'FINAL SUMMARY - TCN MODEL':^80}")
    print(f"{'#' * 80}")
    print(f"\n[Files Saved to: {save_dir}]")
    print(f"\nSaved Files:")
    print(f"  - TCN_yearly_metrics_point.json")
    print(f"  - TCN_yearly_metrics_interval.json")
    print(f"  - TCN_yearly_metrics_probabilistic.json")
    print(f"  - TCN_seasonal_metrics_*.json")
    print(f"  - TCN_monthly_metrics_*.json")
    print(f"  - TCN_hourly_metrics_*.json")
    print(f"  - TCN_test_*.npy")
    print(f"  - TCN_interval_visualization_data.npz")

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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== TCN MODEL TESTING =====")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # 设置参数
    H, W = 48, 96
    batch_size = 4

    # 加载测试数据集
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading test dataset...")
    test_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\test.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test dataset loaded: {len(test_ds)} samples")

    # 初始化模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    model = TCNWindSpeedPredictor(
        H, W,
        tigge_features=10,
        dropout_rate=0.1,
        tcn_channels=[32, 64, 96],
        kernel_size=3,
        output_dim=128
    ).to(device)

    # 加载模型权重
    model_path = r"E:\yym2\qixiang\Obs_PDF\src\checkpoints_TCN_task2\best_model_TCN_task2.pth"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully!")

    # 设置保存目录
    save_dir = r"E:\yym2\qixiang\Obs_PDF\src\resluts_task2\TCN"

    # 运行测试
    results = test_model(model, test_loader, device, save_dir)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== ALL TESTING COMPLETED =====")