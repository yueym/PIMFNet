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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    def __init__(self, in_channels=18, dropout_rate=0.2927417949886402):  # 修改：18 = 10+3+5
        super(ResNetCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 56, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(56)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, 56, kernel_size=1)

        # 新增：多尺度特征融合
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_emb):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out) + time_emb
        out = self.relu(out)

        # 新增：多尺度特征融合
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

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=240, input_dim=10, dropout_rate=0.2927417949886402):  # 修改：input_dim=10
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, h, x, time_features):
        input = torch.cat([h, x, time_features], dim=-1)
        return self.net(input)


class LTC(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=240, output_dim=240, seq_len=4, dt=6.0,
                 dropout_rate=0.2927417949886402):  # 修改：input_dim=10
        super(LTC, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dt = dt
        self.ode_func = ODEFunc(hidden_dim, input_dim, dropout_rate)

        # 新增：特征感知记忆衰减
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, resnet_out, ltc_out):
        fused = torch.cat([resnet_out, ltc_out], dim=1)
        gate = self.gate(fused)
        output = gate * fused
        return output

# ==================== 完整的 WindSpeedPredictor 类 ====================
class WindSpeedPredictor(nn.Module):


    def __init__(self, H, W, tigge_features=10, dropout_rate=0.2927417949886402,
                 ltc_hidden_dim=240, cbam_reduction=8):
        super(WindSpeedPredictor, self).__init__()
        self.H = H
        self.W = W

        # 空间特征提取：ResNet + CBAM
        # 输入通道 = 10(TIGGE) + 3(DEM) + 5(交互) = 18
        self.resnet = ResNetCBAM(
            in_channels=tigge_features + 3 + 5,
            dropout_rate=dropout_rate
        )

        # 时序特征提取：LTC
        self.ltc = LTC(
            input_dim=tigge_features,
            hidden_dim=ltc_hidden_dim,
            output_dim=ltc_hidden_dim,
            dropout_rate=dropout_rate
        )

        # 特征融合
        self.gated_fusion = GatedFusion(56, ltc_hidden_dim)

        # 多任务输出头
        self.multi_task_head = MultiTaskHead(
            56 + ltc_hidden_dim,
            dropout_rate=dropout_rate
        )

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 56)
        )

        # 初始化时间嵌入层
        for m in self.time_embed:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] Total parameters: {total_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,}")

    def forward(self, tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq):

        b = tigge_spatial.size(0)

        # 拼接空间输入
        spatial_input = torch.cat([tigge_spatial, dem_spatial, interaction_spatial], dim=1)

        # 时间嵌入
        time_emb = self.time_embed(time_features_t).view(b, 56, 1, 1)

        # ResNet空间特征提取
        resnet_out = self.resnet(spatial_input, time_emb)

        # LTC时序特征提取
        ltc_out = self.ltc(tigge_seq, time_features_seq)

        # 门控融合
        fused = self.gated_fusion(resnet_out, ltc_out)

        # 多任务输出
        point_pred, mu, sigma, interval_pred = self.multi_task_head(fused)

        return point_pred, mu, sigma, interval_pred


class WindDataset(Dataset):
    def __init__(self, ds_path, H=48, W=96, seq_len=4):
        self.H = H
        self.W = W
        self.seq_len = seq_len
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {ds_path}")
        self.ds = xr.open_dataset(ds_path, cache=False)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset variables: {list(self.ds.data_vars.keys())}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset dimensions: {dict(self.ds.dims)}")

        # 修改：验证新数据集结构
        assert 'tigge_features' in self.ds.data_vars, "缺少 tigge_features"
        assert 'dem_features' in self.ds.data_vars, "缺少 dem_features"
        assert 'interaction_features' in self.ds.data_vars, "缺少 interaction_features"
        assert 'time_features' in self.ds.data_vars, "缺少 time_features"
        assert 'target' in self.ds.data_vars, "缺少 target"

        # 打印特征范围
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

        # 修改：时间特征标准化（与原代码一致）
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Normalizing time features")
        time_data = self.ds['time_features'].values
        self.time_scaler = StandardScaler()
        normalized_time = self.time_scaler.fit_transform(time_data)

        self.ds['time_features_normalized'] = xr.DataArray(
            normalized_time,
            dims=self.ds['time_features'].dims,
            coords={'sample': self.ds['time_features'].coords['sample']}
        )

        # 修改：重构时空索引（从平坦样本恢复时空结构）
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

        # 修改：加载TIGGE序列（仅用于LTC）
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

        # 修改：加载当前时刻的空间数据
        time_t = self.time_points[t]
        mask_t = self.ds.time == time_t

        tigge_spatial = self.ds['tigge_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        dem_spatial = self.ds['dem_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        interaction_spatial = self.ds['interaction_features'].sel(sample=mask_t).values.reshape(self.H, self.W, -1)
        target = self.ds['target'].sel(sample=mask_t).values.reshape(self.H, self.W)

        time_features_t = time_features_seq[-1]

        return {
            'tigge_spatial': torch.from_numpy(tigge_spatial).float().permute(2, 0, 1),  # (10, H, W)
            'dem_spatial': torch.from_numpy(dem_spatial).float().permute(2, 0, 1),  # (3, H, W)
            'interaction_spatial': torch.from_numpy(interaction_spatial).float().permute(2, 0, 1),  # (5, H, W)
            'tigge_seq': torch.from_numpy(tigge_seq).float().permute(0, 3, 1, 2),  # (seq_len, 10, H, W)
            'time_features_t': torch.from_numpy(time_features_t).float(),  # (5,)
            'time_features_seq': torch.from_numpy(time_features_seq).float(),  # (seq_len, 5)
            'target': torch.from_numpy(target).float()  # (H, W)
        }

def visualize_predictions(model, dataloader, device, save_dir='visualizations', num_samples=5):
    """可视化预测结果：概率密度曲线和区间预测"""
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

            # 前向传播
            point_pred, mu, sigma, interval_pred = model(
                tigge_spatial, dem_spatial, interaction_spatial,
                tigge_seq, time_features_t, time_features_seq
            )

            # 选择第一个样本的中心点进行可视化
            h_center, w_center = 24, 48

            target_val = target[0, h_center, w_center].cpu().numpy()
            point_val = point_pred[0, h_center, w_center].cpu().numpy()
            mu_val = mu[0, h_center, w_center].cpu().numpy()
            sigma_val = sigma[0, h_center, w_center].cpu().numpy()
            q_vals = interval_pred[0, :, h_center, w_center].cpu().numpy()

            # 1. 概率密度曲线
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：概率密度
            x_range = np.linspace(max(0, mu_val - 4 * sigma_val),
                                  min(1, mu_val + 4 * sigma_val), 200)
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

            # 右图：区间预测
            x_pos = [0]
            axes[1].errorbar(x_pos, [point_val],
                             yerr=[[point_val - q_vals[0]], [q_vals[3] - point_val]],
                             fmt='o', markersize=8, capsize=10, capthick=2,
                             label='95% CI', color='blue', linewidth=2)
            axes[1].errorbar(x_pos, [point_val],
                             yerr=[[point_val - q_vals[1]], [q_vals[2] - point_val]],
                             fmt='o', markersize=6, capsize=8, capthick=2,
                             label='50% CI (IQR)', color='green', linewidth=2)
            axes[1].scatter(x_pos, [target_val], color='red', s=100,
                            marker='*', label=f'True: {target_val:.3f}', zorder=5)
            axes[1].scatter(x_pos, [point_val], color='blue', s=80,
                            marker='o', label=f'Point: {point_val:.3f}', zorder=4)

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

def train_model(model, train_loader, val_loader, device,
                epochs=12, learning_rate=0.0003323304206226791, weight_decay=0.000048287152161792064,
                patience=5, accumulation_steps=4):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training process")


    criterion = MultiTaskLoss(alpha=1.7000000000000002, beta=0.2, gamma=0.27)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    best_val_loss = float('inf')

    # 修改：记录所有指标
    train_losses = []
    val_losses = []
    train_mae, train_rmse, train_r2 = [], [], []
    val_mae, val_rmse, val_r2 = [], [], []
    train_crps, val_crps = [], []
    train_cp_95, train_cp_50 = [], []
    val_cp_95, val_cp_50 = [], []
    train_mwp_95, train_mwp_50 = [], []
    val_mwp_95, val_mwp_50 = [], []
    train_mc_95, train_mc_50 = [], []
    val_mc_95, val_mc_50 = [], []

    patience_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint directory created")

    # 修改：调试第一个batch
    first_batch = next(iter(train_loader))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_spatial shape: {first_batch['tigge_spatial'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - dem_spatial shape: {first_batch['dem_spatial'].shape}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - interaction_spatial shape: {first_batch['interaction_spatial'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - tigge_seq shape: {first_batch['tigge_seq'].shape}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_t shape: {first_batch['time_features_t'].shape}")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - time_features_seq shape: {first_batch['time_features_seq'].shape}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug - target shape: {first_batch['target'].shape}")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0
        batch_count = 0

        # 修改：收集所有任务的预测结果
        train_point_preds, train_mus, train_sigmas, train_interval_preds = [], [], [], []
        train_targets = []

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            try:
                # 修改：加载新的输入
                tigge_spatial = batch['tigge_spatial'].to(device)
                dem_spatial = batch['dem_spatial'].to(device)
                interaction_spatial = batch['interaction_spatial'].to(device)
                tigge_seq = batch['tigge_seq'].to(device)
                time_features_t = batch['time_features_t'].to(device)
                time_features_seq = batch['time_features_seq'].to(device)
                target = batch['target'].to(device)

                with autocast():
                    # 修改：多任务输出
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

                # ========== 修改5：增强梯度监控 ==========
                scaler.scale(loss / accumulation_steps).backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)

                    # 计算总梯度范数（在裁剪前）
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    # 检查梯度异常
                    if np.isnan(total_norm) or np.isinf(total_norm):
                        print(f"[CRITICAL] 梯度为NaN/Inf! 跳过batch {batch_idx}")
                        optimizer.zero_grad()
                        continue

                    if total_norm > 50.0:  # 梯度爆炸阈值
                        print(f"[WARNING] 梯度过大: {total_norm:.2f}, 跳过batch {batch_idx}")
                        optimizer.zero_grad()
                        continue

                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                    # 定期打印梯度信息
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

   
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing training metrics")

        train_point_preds_tensor = torch.cat(train_point_preds)  # (N, H, W) - float32
        train_mus_tensor = torch.cat(train_mus)  # (N, H, W) - float32
        train_sigmas_tensor = torch.cat(train_sigmas)  # (N, H, W) - float32
        train_interval_preds_tensor = torch.cat(train_interval_preds)  # (N, 4, H, W) - float32
        train_targets_tensor = torch.cat(train_targets)  # (N, H, W) - float32

        # 展平用于点预测指标
        train_point_preds_flat = train_point_preds_tensor.numpy().flatten()
        train_mus_flat = train_mus_tensor.numpy().flatten()
        train_sigmas_flat = train_sigmas_tensor.numpy().flatten()
        train_targets_flat = train_targets_tensor.numpy().flatten()

        # 点预测指标
        train_mae_val = np.mean(np.abs(train_point_preds_flat - train_targets_flat))
        train_rmse_val = np.sqrt(np.mean((train_point_preds_flat - train_targets_flat) ** 2))
        train_mean = np.mean(train_targets_flat)
        ss_tot = np.sum((train_targets_flat - train_mean) ** 2)
        ss_res = np.sum((train_targets_flat - train_point_preds_flat) ** 2)
        train_r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # CRPS指标
        train_crps_val = compute_crps(
            torch.from_numpy(train_mus_flat),
            torch.from_numpy(train_sigmas_flat),
            torch.from_numpy(train_targets_flat)
        ).item()

        train_interval_metrics = compute_interval_metrics(
            train_interval_preds_tensor,  # (N, 4, H, W) - float32
            train_targets_tensor,  # (N, H, W) - float32
            train_point_preds_tensor  # ✅ 新增：传入点预测 (N, H, W) - float32
        )

        # 记录指标
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

        # ==================== 验证阶段 ====================
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting validation")
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        val_point_preds, val_mus, val_sigmas, val_interval_preds = [], [], [], []
        val_targets = []

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

        # ========== 修改：计算验证指标（添加point_pred的拼接和传递） ==========
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Computing validation metrics")

        val_point_preds_tensor = torch.cat(val_point_preds)  # (N, H, W) - float32
        val_mus_tensor = torch.cat(val_mus)  # (N, H, W) - float32
        val_sigmas_tensor = torch.cat(val_sigmas)  # (N, H, W) - float32
        val_interval_preds_tensor = torch.cat(val_interval_preds)  # (N, 4, H, W) - float32
        val_targets_tensor = torch.cat(val_targets)  # (N, H, W) - float32

        # 展平用于点预测指标
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
            val_interval_preds_tensor,  # (N, 4, H, W) - float32
            val_targets_tensor,  # (N, H, W) - float32
            val_point_preds_tensor  # ✅ 新增：传入点预测 (N, H, W) - float32
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
            torch.save(model.state_dict(), 'checkpoints/best_model_task_two.pth')
            print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Model saved with val loss: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Early stopping at epoch {epoch + 1}')
                break

        # 修改：保存所有指标
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
        }, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth')

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        remaining_epochs = epochs - (epoch + 1)
        est_remaining_time = remaining_epochs * epoch_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1} completed in {epoch_time / 60:.2f} minutes")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Estimated time remaining: {est_remaining_time / 3600:.2f} hours")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training completed in {total_time / 3600:.2f} hours")

    # 修改：生成所有可视化图表
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating visualization plots")

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=300)
    plt.close()

    # 2. MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_mae) + 1), train_mae, label='Train MAE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_mae) + 1), val_mae, label='Validation MAE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Mean Absolute Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_curve.png', dpi=300)
    plt.close()

    # 3. RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE', color='blue', linewidth=2)
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, label='Validation RMSE', color='orange', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Root Mean Squared Error', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_curve.png', dpi=300)
    plt.close()

   
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All metric curves saved")

    # 新增：9. 生成预测可视化
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction visualizations")
    visualize_predictions(model, val_loader, device, save_dir='visualizations', num_samples=5)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training pipeline completed successfully!")


# ==================== 修改：主程序 ====================
if __name__ == "__main__":
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Program started")

    set_seed(42)

    H, W = 48, 96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Setting up training parameters")
    batch_size = 8
    epochs = 12
    learning_rate = 0.0003323304206226791
    weight_decay = 0.000048287152161792064
    patience = 5
    accumulation_steps = 4



    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")

    # 修改：模型参数调整
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model")
    model = WindSpeedPredictor(
        H, W,
        tigge_features=10,  # 修改：10个TIGGE特征
        dropout_rate=0.2927417949886402,
        ltc_hidden_dim=240,
        cbam_reduction=8
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
