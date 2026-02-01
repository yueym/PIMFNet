import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import pandas as pd
import json
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast


# ==================== 模型定义（与训练代码保持一致）====================
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        fused = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        global_feat = self.global_pool(x)
        global_feat = self.fc(global_feat.view(x.size(0), -1))
        global_att = self.sigmoid(global_feat).view(x.size(0), -1, 1, 1)
        x = x + x * global_att
        return x


class ResNetCBAM(nn.Module):
    def __init__(self, in_channels=18, dropout_rate=0.2927417949886402):
        super(ResNetCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 56, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(56)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, 56, kernel_size=1)
        self.multi_scale = MultiScaleFeatureFusion(56)
        self.conv2 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(56)
        self.conv3 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(56)
        self.conv4 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(56)
        self.conv5 = nn.Conv2d(56, 56, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(56)
        self.cbam = CBAM(56, reduction=8)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, time_emb):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out) + time_emb
        out = self.relu(out)
        out = self.multi_scale(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv4(out)
        out = self.bn4(out)
        out = out + identity
        out = self.relu(out)
        identity = out
        out = self.conv5(out)
        out = self.bn5(out)
        out = out + identity
        out = self.relu(out)
        out = self.dropout(out)
        out = self.cbam(out)
        return out


class FeatureAwareMemoryDecay(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(FeatureAwareMemoryDecay, self).__init__()
        self.decay_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, h, x):
        decay_rate = self.decay_net(torch.cat([h, x], dim=-1))
        h_decayed = h * decay_rate
        return h_decayed


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=240, input_dim=10, dropout_rate=0.2927417949886402):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 5, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, hidden_dim)
        )

    def forward(self, h, x, time_features):
        input = torch.cat([h, x, time_features], dim=-1)
        return self.net(input)


class LTC(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=240, output_dim=240, seq_len=4, dt=6.0,
                 dropout_rate=0.2927417949886402):
        super(LTC, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dt = dt
        self.ode_func = ODEFunc(hidden_dim, input_dim, dropout_rate)
        self.memory_decay = FeatureAwareMemoryDecay(hidden_dim, input_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq, time_features_seq):
        b, t, c, H, W = x_seq.shape
        x_seq = x_seq.permute(0, 3, 4, 1, 2).reshape(b * H * W, self.seq_len, -1)
        time_features_seq = time_features_seq.unsqueeze(1).unsqueeze(2).repeat(1, H, W, 1, 1).reshape(b * H * W,
                                                                                                      self.seq_len, 5)
        h = torch.zeros(b * H * W, self.hidden_dim).to(x_seq.device)
        dt = self.dt * 0.1
        for k in range(self.seq_len):
            x_k = x_seq[:, k, :]
            t_k = time_features_seq[:, k, :]
            h = self.memory_decay(h, x_k)
            if torch.isnan(h).any():
                h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
            k1 = self.ode_func(h, x_k, t_k)
            k2 = self.ode_func(h + 0.5 * dt * k1, x_k, t_k)
            k3 = self.ode_func(h + 0.5 * dt * k2, x_k, t_k)
            k4 = self.ode_func(h + dt * k3, x_k, t_k)
            k1 = torch.clamp(k1, -10, 10)
            k2 = torch.clamp(k2, -10, 10)
            k3 = torch.clamp(k3, -10, 10)
            k4 = torch.clamp(k4, -10, 10)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            h = torch.clamp(h, -100, 100)
        out = self.output_layer(h)
        out = out.reshape(b, H, W, -1).permute(0, 3, 1, 2)
        return out


class GatedFusion(nn.Module):
    def __init__(self, C1, C2):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(C1 + C2, (C1 + C2) // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d((C1 + C2) // 2, C1 + C2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, resnet_out, ltc_out):
        fused = torch.cat([resnet_out, ltc_out], dim=1)
        gate = self.gate(fused)
        output = gate * fused
        return output


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2927417949886402):
        super(MultiTaskHead, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(512, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.point_head = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.prob_mu = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.prob_log_sigma = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 1, kernel_size=1)
        )
        self.interval_head = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(192, 4, kernel_size=1),
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


class WindSpeedPredictor(nn.Module):
    def __init__(self, H, W, tigge_features=10, dropout_rate=0.2927417949886402,
                 ltc_hidden_dim=240, cbam_reduction=8):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W
        self.resnet = ResNetCBAM(in_channels=tigge_features + 3 + 5, dropout_rate=dropout_rate)
        self.ltc = LTC(input_dim=tigge_features, hidden_dim=ltc_hidden_dim, output_dim=ltc_hidden_dim,
                       dropout_rate=dropout_rate)
        self.gated_fusion = GatedFusion(56, ltc_hidden_dim)
        self.multi_task_head = MultiTaskHead(56 + ltc_hidden_dim, dropout_rate=dropout_rate)
        self.time_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 56)
        )

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):
        b = tigge_spatial.size(0)
        spatial_input = torch.cat([tigge_spatial, dem_spatial, interaction_spatial], dim=1)
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)
        resnet_out = self.resnet(spatial_input, time_emb)
        ltc_out = self.ltc(tigge_seq, time_features_seq)
        fused = self.gated_fusion(resnet_out, ltc_out)
        point_pred, mu, sigma, interval_pred = self.multi_task_head(fused)
        return point_pred, mu, sigma, interval_pred


# ==================== 数据集类 ====================
class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

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


# ==================== 点预测指标计算（8个指标）====================
def calculate_point_metrics(pred, target, tigge_wind=None):
    """
    计算点预测指标
    pred, target, tigge_wind: numpy arrays (已反标准化)
    """
    pred = pred.flatten()
    target = target.flatten()

    # 预测指标
    FA_pred = np.mean(np.abs(pred - target) < 1) * 100
    RMSE_pred = np.sqrt(np.mean((pred - target) ** 2))
    MAE_pred = np.mean(np.abs(pred - target))
    mean_target = np.mean(target)
    rRMSE_pred = (RMSE_pred / mean_target) * 100 if mean_target > 0 else 0
    rMAE_pred = (MAE_pred / mean_target) * 100 if mean_target > 0 else 0

    # 相关系数
    if np.std(pred) > 0 and np.std(target) > 0:
        R_pred = np.corrcoef(pred, target)[0, 1]
    else:
        R_pred = 0

    # R²
    ss_tot = np.sum((target - mean_target) ** 2)
    ss_res = np.sum((target - pred) ** 2)
    R2_pred = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # MAPE
    epsilon = 1e-8
    target_safe = np.where(np.abs(target) < epsilon, epsilon, target)
    MAPE_pred = np.mean(np.abs((pred - target) / target_safe)) * 100

    result = {
        'FA_pred': float(FA_pred),
        'RMSE_pred': float(RMSE_pred),
        'MAE_pred': float(MAE_pred),
        'rRMSE_pred': float(rRMSE_pred),
        'rMAE_pred': float(rMAE_pred),
        'R_pred': float(R_pred),
        'R2_pred': float(R2_pred),
        'MAPE_pred': float(MAPE_pred)
    }

    # 如果提供了TIGGE风速，也计算TIGGE的指标
    if tigge_wind is not None:
        tigge_wind = tigge_wind.flatten()
        FA_tigge = np.mean(np.abs(tigge_wind - target) < 1) * 100
        RMSE_tigge = np.sqrt(np.mean((tigge_wind - target) ** 2))
        MAE_tigge = np.mean(np.abs(tigge_wind - target))
        rRMSE_tigge = (RMSE_tigge / mean_target) * 100 if mean_target > 0 else 0
        rMAE_tigge = (MAE_tigge / mean_target) * 100 if mean_target > 0 else 0

        if np.std(tigge_wind) > 0 and np.std(target) > 0:
            R_tigge = np.corrcoef(tigge_wind, target)[0, 1]
        else:
            R_tigge = 0

        ss_res_tigge = np.sum((target - tigge_wind) ** 2)
        R2_tigge = 1 - (ss_res_tigge / ss_tot) if ss_tot > 0 else 0
        MAPE_tigge = np.mean(np.abs((tigge_wind - target) / target_safe)) * 100

        result.update({
            'FA_tigge': float(FA_tigge),
            'RMSE_tigge': float(RMSE_tigge),
            'MAE_tigge': float(MAE_tigge),
            'rRMSE_tigge': float(rRMSE_tigge),
            'rMAE_tigge': float(rMAE_tigge),
            'R_tigge': float(R_tigge),
            'R2_tigge': float(R2_tigge),
            'MAPE_tigge': float(MAPE_tigge)
        })

    return result


# ==================== 区间预测指标计算（4个指标：CP, MWP, MC, PINAW）====================
def calculate_interval_metrics(interval_pred, target, point_pred):
    """
    计算区间预测指标
    interval_pred: (N, 4, H, W) - 4个分位数 [0.025, 0.25, 0.75, 0.975]
    target: (N, H, W)
    point_pred: (N, H, W)
    所有数据已反标准化
    """
    q_0025 = interval_pred[:, 0, :, :].flatten()
    q_025 = interval_pred[:, 1, :, :].flatten()
    q_075 = interval_pred[:, 2, :, :].flatten()
    q_0975 = interval_pred[:, 3, :, :].flatten()
    target_flat = target.flatten()
    point_pred_flat = point_pred.flatten()

    # CP (Coverage Probability) - 公式(29)
    coverage_95 = np.mean((target_flat >= q_0025) & (target_flat <= q_0975))
    coverage_50 = np.mean((target_flat >= q_025) & (target_flat <= q_075))

    # MWP (Mean Width Percentage) - 公式(30)
    point_pred_safe = np.maximum(np.abs(point_pred_flat), 1e-3)
    width_95 = q_0975 - q_0025
    width_50 = q_075 - q_025
    mwp_95 = np.mean(width_95 / point_pred_safe)
    mwp_50 = np.mean(width_50 / point_pred_safe)

    # MC (Mean Coverage) - 公式(31)
    mc_95 = mwp_95 / max(coverage_95, 1e-6)
    mc_50 = mwp_50 / max(coverage_50, 1e-6)

    # PINAW (Prediction Interval Normalized Average Width)
    target_range = np.max(target_flat) - np.min(target_flat)
    if target_range > 0:
        pinaw_95 = np.mean(width_95) / target_range
        pinaw_50 = np.mean(width_50) / target_range
    else:
        pinaw_95 = 0
        pinaw_50 = 0

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


# ==================== 概率预测指标计算（CRPS）====================
def calculate_crps(mu, sigma, target):
    """
    计算CRPS (Continuous Ranked Probability Score)
    mu, sigma, target: numpy arrays (已反标准化)
    """
    mu = mu.flatten()
    sigma = sigma.flatten()
    target = target.flatten()

    # 限制sigma范围
    sigma = np.clip(sigma, 0.01, None)

    # 标准化误差
    z = (target - mu) / sigma
    z = np.clip(z, -5.0, 5.0)

    # 标准正态分布CDF和PDF
    from scipy.stats import norm
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # CRPS公式
    crps = sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1.0 / np.sqrt(np.pi))
    crps_mean = np.mean(np.abs(crps))

    return float(crps_mean)


# ==================== 测试函数（完整修复版 v4 - 分开保存Proposed和TIGGE指标）====================
def test_model(model, test_loader, device, save_dir):
    """
    测试模型并保存所有指标和预测结果
    Proposed和TIGGE的指标分开保存到不同文件
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting model testing...")

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 加载标准化器
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading scalers...")
    scaler_target = joblib.load(r'E:\yym2\qixiang\Obs_PDF\final\target_scaler.pkl')
    scaler_tigge_wind = joblib.load(
        r'E:\yym2\qixiang\Obs_PDF\processed_data_v4_sequences\scalers\scaler_tigge_wind.pkl')

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler type: {type(scaler_target).__name__}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIGGE wind scaler type: {type(scaler_tigge_wind).__name__}")

    # 根据scaler类型获取反标准化参数
    if hasattr(scaler_target, 'data_min_'):
        target_data_min = scaler_target.data_min_[0]
        target_range = 1 / scaler_target.scale_[0]
        target_scaler_type = 'MinMaxScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: MinMaxScaler, data_min={target_data_min:.4f}, range={target_range:.4f}")
    elif hasattr(scaler_target, 'center_'):
        target_center = scaler_target.center_[0]
        target_scale = scaler_target.scale_[0]
        target_scaler_type = 'RobustScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: RobustScaler, center={target_center:.4f}, scale={target_scale:.4f}")
    elif hasattr(scaler_target, 'mean_'):
        target_mean = scaler_target.mean_[0]
        target_std = scaler_target.scale_[0]
        target_scaler_type = 'StandardScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Target scaler: StandardScaler, mean={target_mean:.4f}, std={target_std:.4f}")
    else:
        raise ValueError(f"Unknown target scaler type: {type(scaler_target)}")

    if hasattr(scaler_tigge_wind, 'data_min_'):
        tigge_data_min = scaler_tigge_wind.data_min_[0]
        tigge_range = 1 / scaler_tigge_wind.scale_[0]
        tigge_scaler_type = 'MinMaxScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIGGE scaler: MinMaxScaler, data_min={tigge_data_min:.4f}, range={tigge_range:.4f}")
    elif hasattr(scaler_tigge_wind, 'center_'):
        tigge_center = scaler_tigge_wind.center_[0]
        tigge_scale = scaler_tigge_wind.scale_[0]
        tigge_scaler_type = 'RobustScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIGGE scaler: RobustScaler, center={tigge_center:.4f}, scale={tigge_scale:.4f}")
    elif hasattr(scaler_tigge_wind, 'mean_'):
        tigge_mean = scaler_tigge_wind.mean_[0]
        tigge_std = scaler_tigge_wind.scale_[0]
        tigge_scaler_type = 'StandardScaler'
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIGGE scaler: StandardScaler, mean={tigge_mean:.4f}, std={tigge_std:.4f}")
    else:
        raise ValueError(f"Unknown TIGGE wind scaler type: {type(scaler_tigge_wind)}")

    # 定义反标准化函数
    def inverse_transform_target(data):
        if target_scaler_type == 'MinMaxScaler':
            return (data * target_range) + target_data_min
        elif target_scaler_type == 'RobustScaler':
            return (data * target_scale) + target_center
        elif target_scaler_type == 'StandardScaler':
            return (data * target_std) + target_mean

    def inverse_transform_tigge(data):
        if tigge_scaler_type == 'MinMaxScaler':
            return (data * tigge_range) + tigge_data_min
        elif tigge_scaler_type == 'RobustScaler':
            return (data * tigge_scale) + tigge_center
        elif tigge_scaler_type == 'StandardScaler':
            return (data * tigge_std) + tigge_mean

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
    all_tigge_wind = []
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
                all_tigge_wind.append(tigge_spatial[:, 0, :, :].cpu().float().numpy())

                actual_indices = test_loader.dataset.sample_indices[
                                 batch_idx * test_loader.batch_size:
                                 (batch_idx + 1) * test_loader.batch_size]
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
    all_tigge_wind = np.concatenate(all_tigge_wind, axis=0)
    all_dates = np.array(all_dates[:len(all_point_preds)])

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total samples: {len(all_point_preds)}")

    # ==================== 反标准化 ====================
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inverse transforming predictions...")

    all_point_preds_orig = inverse_transform_target(all_point_preds)
    all_targets_orig = inverse_transform_target(all_targets)
    all_tigge_wind_orig = inverse_transform_tigge(all_tigge_wind)
    all_mus_orig = inverse_transform_target(all_mus)
    all_sigmas_orig = inverse_transform_sigma(all_sigmas)
    all_interval_preds_orig = inverse_transform_target(all_interval_preds)

    # 裁剪到合理范围
    all_point_preds_orig = np.clip(all_point_preds_orig, 0, 100)
    all_targets_orig = np.clip(all_targets_orig, 0, 100)
    all_tigge_wind_orig = np.clip(all_tigge_wind_orig, 0, 100)
    all_mus_orig = np.clip(all_mus_orig, 0, 100)
    all_sigmas_orig = np.clip(all_sigmas_orig, 0.01, 50)
    all_interval_preds_orig = np.clip(all_interval_preds_orig, 0, 100)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] After inverse transform:")
    print(f"  Point predictions range: [{all_point_preds_orig.min():.4f}, {all_point_preds_orig.max():.4f}]")
    print(f"  Targets range: [{all_targets_orig.min():.4f}, {all_targets_orig.max():.4f}]")
    print(f"  TIGGE wind range: [{all_tigge_wind_orig.min():.4f}, {all_tigge_wind_orig.max():.4f}]")
    print(f"  Mu range: [{all_mus_orig.min():.4f}, {all_mus_orig.max():.4f}]")
    print(f"  Sigma range: [{all_sigmas_orig.min():.4f}, {all_sigmas_orig.max():.4f}]")
    print(f"  Interval range: [{all_interval_preds_orig.min():.4f}, {all_interval_preds_orig.max():.4f}]")

    # ==================== 保存原始预测结果 ====================
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving prediction results...")

    np.save(os.path.join(save_dir, 'proposed_test_point_preds.npy'), all_point_preds_orig)
    np.save(os.path.join(save_dir, 'proposed_test_targets.npy'), all_targets_orig)
    np.save(os.path.join(save_dir, 'proposed_test_tigge_wind.npy'), all_tigge_wind_orig)
    np.save(os.path.join(save_dir, 'proposed_test_dates.npy'), all_dates)
    np.save(os.path.join(save_dir, 'proposed_test_mus.npy'), all_mus_orig)
    np.save(os.path.join(save_dir, 'proposed_test_sigmas.npy'), all_sigmas_orig)
    np.save(os.path.join(save_dir, 'proposed_test_intervals.npy'), all_interval_preds_orig)

    # ==================== 辅助函数 ====================
    def to_numpy_mask(mask):
        if hasattr(mask, 'values'):
            return mask.values
        return np.array(mask)

    def calculate_point_metrics_proposed(pred, target):
        """计算Proposed模型的点预测指标（8个）"""
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

    def calculate_point_metrics_tigge(tigge_wind, target):
        """计算TIGGE的点预测指标（8个）"""
        tigge_wind = tigge_wind.flatten()
        target = target.flatten()

        FA = np.mean(np.abs(tigge_wind - target) < 1) * 100
        RMSE = np.sqrt(np.mean((tigge_wind - target) ** 2))
        MAE = np.mean(np.abs(tigge_wind - target))
        mean_target = np.mean(target)
        rRMSE = (RMSE / mean_target) * 100 if mean_target > 0 else 0
        rMAE = (MAE / mean_target) * 100 if mean_target > 0 else 0

        if np.std(tigge_wind) > 0 and np.std(target) > 0:
            R = np.corrcoef(tigge_wind, target)[0, 1]
        else:
            R = 0

        ss_tot = np.sum((target - mean_target) ** 2)
        ss_res = np.sum((target - tigge_wind) ** 2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        epsilon = 1e-8
        target_safe = np.where(np.abs(target) < epsilon, epsilon, target)
        MAPE = np.mean(np.abs((tigge_wind - target) / target_safe)) * 100

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

    def compute_all_metrics_for_subset(mask, subset_name):
        """计算某个子集的所有指标，分开返回Proposed和TIGGE"""
        mask = to_numpy_mask(mask)
        if np.sum(mask) == 0:
            print(f"[WARNING] No samples for {subset_name}")
            return None

        subset_preds = all_point_preds_orig[mask]
        subset_targets = all_targets_orig[mask]
        subset_tigge = all_tigge_wind_orig[mask]
        subset_mus = all_mus_orig[mask]
        subset_sigmas = all_sigmas_orig[mask]
        subset_intervals = all_interval_preds_orig[mask]

        # Proposed模型的点预测指标
        proposed_point_metrics = calculate_point_metrics_proposed(subset_preds, subset_targets)

        # TIGGE的点预测指标
        tigge_point_metrics = calculate_point_metrics_tigge(subset_tigge, subset_targets)

        # Proposed模型的区间预测指标
        interval_metrics = calculate_interval_metrics(subset_intervals, subset_targets, subset_preds)

        # Proposed模型的概率预测指标(CRPS)
        crps = calculate_crps(subset_mus, subset_sigmas, subset_targets)

        return {
            'proposed_point': proposed_point_metrics,
            'tigge_point': tigge_point_metrics,
            'proposed_interval': interval_metrics,
            'proposed_probabilistic': {'CRPS': crps}
        }

    def print_full_metrics(metrics, period_name):
        """打印完整的所有指标"""
        print(f"\n{'=' * 70}")
        print(f"[{period_name}] COMPLETE METRICS")
        print(f"{'=' * 70}")

        # Proposed点预测指标 (8个)
        print(f"\n[{period_name}] Point Prediction - Proposed Model (8 metrics):")
        print(f"  {'Metric':<12} {'Value':>12}")
        print(f"  {'-' * 24}")
        for metric in ['FA', 'RMSE', 'MAE', 'rRMSE', 'rMAE', 'R', 'R2', 'MAPE']:
            if metric in metrics['proposed_point']:
                print(f"  {metric:<12} {metrics['proposed_point'][metric]:>12.4f}")

        # TIGGE点预测指标 (8个)
        print(f"\n[{period_name}] Point Prediction - TIGGE (8 metrics):")
        print(f"  {'Metric':<12} {'Value':>12}")
        print(f"  {'-' * 24}")
        for metric in ['FA', 'RMSE', 'MAE', 'rRMSE', 'rMAE', 'R', 'R2', 'MAPE']:
            if metric in metrics['tigge_point']:
                print(f"  {metric:<12} {metrics['tigge_point'][metric]:>12.4f}")

        # Proposed区间预测指标 (8个: 4个95%CI + 4个50%CI)
        print(f"\n[{period_name}] Interval Prediction - Proposed Model (8 metrics):")
        print(f"  {'Metric':<12} {'95% CI':>12} {'50% CI':>12}")
        print(f"  {'-' * 36}")
        print(
            f"  {'CP':<12} {metrics['proposed_interval']['CP_95']:>12.4f} {metrics['proposed_interval']['CP_50']:>12.4f}")
        print(
            f"  {'MWP':<12} {metrics['proposed_interval']['MWP_95']:>12.4f} {metrics['proposed_interval']['MWP_50']:>12.4f}")
        print(
            f"  {'MC':<12} {metrics['proposed_interval']['MC_95']:>12.4f} {metrics['proposed_interval']['MC_50']:>12.4f}")
        print(
            f"  {'PINAW':<12} {metrics['proposed_interval']['PINAW_95']:>12.4f} {metrics['proposed_interval']['PINAW_50']:>12.4f}")

        # Proposed概率预测指标 (1个)
        print(f"\n[{period_name}] Probabilistic Prediction - Proposed Model (1 metric):")
        print(f"  CRPS: {metrics['proposed_probabilistic']['CRPS']:.4f}")

    # 转换日期格式
    dates_pd = pd.to_datetime(all_dates)

    # ========== 1. 年度指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== YEARLY METRICS =====")
    yearly_mask = np.ones(len(all_dates), dtype=bool)
    yearly_metrics = compute_all_metrics_for_subset(yearly_mask, "Yearly")
    print_full_metrics(yearly_metrics, "YEARLY")

    # 分开保存Proposed和TIGGE的年度指标
    with open(os.path.join(save_dir, 'proposed_yearly_metrics_point.json'), 'w') as f:
        json.dump(yearly_metrics['proposed_point'], f, indent=2)
    with open(os.path.join(save_dir, 'tigge_yearly_metrics_point.json'), 'w') as f:
        json.dump(yearly_metrics['tigge_point'], f, indent=2)
    with open(os.path.join(save_dir, 'proposed_yearly_metrics_interval.json'), 'w') as f:
        json.dump(yearly_metrics['proposed_interval'], f, indent=2)
    with open(os.path.join(save_dir, 'proposed_yearly_metrics_probabilistic.json'), 'w') as f:
        json.dump(yearly_metrics['proposed_probabilistic'], f, indent=2)

    # ========== 2. 季节指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== SEASONAL METRICS =====")
    seasons = {
        'Spring': (3, 5),
        'Summer': (6, 8),
        'Autumn': (9, 11),
        'Winter': (12, 2)
    }
    proposed_seasonal_metrics_point = {}
    tigge_seasonal_metrics_point = {}
    proposed_seasonal_metrics_interval = {}
    proposed_seasonal_metrics_probabilistic = {}

    for season, (start_month, end_month) in seasons.items():
        if start_month < end_month:
            mask = (dates_pd.month >= start_month) & (dates_pd.month <= end_month)
        else:
            mask = (dates_pd.month >= start_month) | (dates_pd.month <= end_month)

        mask_np = to_numpy_mask(mask)
        metrics = compute_all_metrics_for_subset(mask_np, season)
        if metrics:
            proposed_seasonal_metrics_point[season] = metrics['proposed_point']
            tigge_seasonal_metrics_point[season] = metrics['tigge_point']
            proposed_seasonal_metrics_interval[season] = metrics['proposed_interval']
            proposed_seasonal_metrics_probabilistic[season] = metrics['proposed_probabilistic']
            print_full_metrics(metrics, season.upper())

    # 分开保存
    with open(os.path.join(save_dir, 'proposed_seasonal_metrics_point.json'), 'w') as f:
        json.dump(proposed_seasonal_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'tigge_seasonal_metrics_point.json'), 'w') as f:
        json.dump(tigge_seasonal_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_seasonal_metrics_interval.json'), 'w') as f:
        json.dump(proposed_seasonal_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_seasonal_metrics_probabilistic.json'), 'w') as f:
        json.dump(proposed_seasonal_metrics_probabilistic, f, indent=2)

    # ========== 3. 月份指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== MONTHLY METRICS =====")
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    proposed_monthly_metrics_point = {}
    tigge_monthly_metrics_point = {}
    proposed_monthly_metrics_interval = {}
    proposed_monthly_metrics_probabilistic = {}

    for month in range(1, 13):
        mask = (dates_pd.month == month)
        mask_np = to_numpy_mask(mask)
        metrics = compute_all_metrics_for_subset(mask_np, months_names[month - 1])
        if metrics:
            proposed_monthly_metrics_point[str(month)] = metrics['proposed_point']
            tigge_monthly_metrics_point[str(month)] = metrics['tigge_point']
            proposed_monthly_metrics_interval[str(month)] = metrics['proposed_interval']
            proposed_monthly_metrics_probabilistic[str(month)] = metrics['proposed_probabilistic']
            print_full_metrics(metrics, months_names[month - 1].upper())

    # 分开保存
    with open(os.path.join(save_dir, 'proposed_monthly_metrics_point.json'), 'w') as f:
        json.dump(proposed_monthly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'tigge_monthly_metrics_point.json'), 'w') as f:
        json.dump(tigge_monthly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_monthly_metrics_interval.json'), 'w') as f:
        json.dump(proposed_monthly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_monthly_metrics_probabilistic.json'), 'w') as f:
        json.dump(proposed_monthly_metrics_probabilistic, f, indent=2)

    # ========== 4. 四个时间点指标 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== HOURLY METRICS (4 time points) =====")
    hours = [0, 6, 12, 18]
    proposed_hourly_metrics_point = {}
    tigge_hourly_metrics_point = {}
    proposed_hourly_metrics_interval = {}
    proposed_hourly_metrics_probabilistic = {}

    for hour in hours:
        mask = (dates_pd.hour == hour)
        mask_np = to_numpy_mask(mask)
        hour_str = f"{hour:02d}:00"
        metrics = compute_all_metrics_for_subset(mask_np, hour_str)
        if metrics:
            proposed_hourly_metrics_point[str(hour)] = metrics['proposed_point']
            tigge_hourly_metrics_point[str(hour)] = metrics['tigge_point']
            proposed_hourly_metrics_interval[str(hour)] = metrics['proposed_interval']
            proposed_hourly_metrics_probabilistic[str(hour)] = metrics['proposed_probabilistic']
            print_full_metrics(metrics, hour_str)

    # 分开保存
    with open(os.path.join(save_dir, 'proposed_hourly_metrics_point.json'), 'w') as f:
        json.dump(proposed_hourly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'tigge_hourly_metrics_point.json'), 'w') as f:
        json.dump(tigge_hourly_metrics_point, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_hourly_metrics_interval.json'), 'w') as f:
        json.dump(proposed_hourly_metrics_interval, f, indent=2)
    with open(os.path.join(save_dir, 'proposed_hourly_metrics_probabilistic.json'), 'w') as f:
        json.dump(proposed_hourly_metrics_probabilistic, f, indent=2)

    # ========== 5. 保存区间可视化数据 ==========
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving interval predictions for visualization...")
    np.savez(
        os.path.join(save_dir, 'proposed_interval_visualization_data.npz'),
        dates=all_dates,
        point_pred=all_point_preds_orig,
        target=all_targets_orig,
        q_0025=all_interval_preds_orig[:, 0, :, :],
        q_025=all_interval_preds_orig[:, 1, :, :],
        q_075=all_interval_preds_orig[:, 2, :, :],
        q_0975=all_interval_preds_orig[:, 3, :, :]
    )

    # ==================== 最终汇总 ====================
    print(f"\n\n{'#' * 80}")
    print(f"{'FINAL SUMMARY - SAVED FILES':^80}")
    print(f"{'#' * 80}")

    print(f"\n[Files Saved to: {save_dir}]")
    print(f"\nProposed Model Files:")
    print(f"  - proposed_yearly_metrics_point.json (8 metrics)")
    print(f"  - proposed_seasonal_metrics_point.json (8 metrics × 4 seasons)")
    print(f"  - proposed_monthly_metrics_point.json (8 metrics × 12 months)")
    print(f"  - proposed_hourly_metrics_point.json (8 metrics × 4 hours)")
    print(f"  - proposed_*_metrics_interval.json (8 metrics each)")
    print(f"  - proposed_*_metrics_probabilistic.json (1 metric each)")

    print(f"\nTIGGE Model Files:")
    print(f"  - tigge_yearly_metrics_point.json (8 metrics)")
    print(f"  - tigge_seasonal_metrics_point.json (8 metrics × 4 seasons)")
    print(f"  - tigge_monthly_metrics_point.json (8 metrics × 12 months)")
    print(f"  - tigge_hourly_metrics_point.json (8 metrics × 4 hours)")

    print(f"\nPrediction Data Files (.npy):")
    print(f"  - proposed_test_point_preds.npy")
    print(f"  - proposed_test_targets.npy")
    print(f"  - proposed_test_tigge_wind.npy")
    print(f"  - proposed_test_dates.npy")
    print(f"  - proposed_test_mus.npy")
    print(f"  - proposed_test_sigmas.npy")
    print(f"  - proposed_test_intervals.npy")

    print(f"\nVisualization Data:")
    print(f"  - proposed_interval_visualization_data.npz")

    print(f"\n{'=' * 80}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing completed successfully!")
    print(f"{'=' * 80}")

    return {
        'yearly': yearly_metrics,
        'seasonal': {
            'proposed_point': proposed_seasonal_metrics_point,
            'tigge_point': tigge_seasonal_metrics_point,
            'proposed_interval': proposed_seasonal_metrics_interval,
            'proposed_probabilistic': proposed_seasonal_metrics_probabilistic
        },
        'monthly': {
            'proposed_point': proposed_monthly_metrics_point,
            'tigge_point': tigge_monthly_metrics_point,
            'proposed_interval': proposed_monthly_metrics_interval,
            'proposed_probabilistic': proposed_monthly_metrics_probabilistic
        },
        'hourly': {
            'proposed_point': proposed_hourly_metrics_point,
            'tigge_point': tigge_hourly_metrics_point,
            'proposed_interval': proposed_hourly_metrics_interval,
            'proposed_probabilistic': proposed_hourly_metrics_probabilistic
        }
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== PROPOSED MODEL TESTING =====")

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    # 加载测试数据
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading test dataset...")
    batch_size = 8
    test_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\test.nc", H, W)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Test dataset loaded: {len(test_ds)} samples")

    # 初始化模型
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model...")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=10,
        dropout_rate=0.2927417949886402,
        ltc_hidden_dim=240,
        cbam_reduction=8
    ).to(device)

    # 加载最佳模型权重
    model_path = r"E:\yym2\qixiang\Obs_PDF\src\checkpoints\best_model_task_two.pth"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully!")

    # 设置保存目录
    save_dir = r"E:\yym2\qixiang\Obs_PDF\src\resluts_task2"

    # 运行测试
    results = test_model(model, test_loader, device, save_dir)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== ALL TESTING COMPLETED =====")