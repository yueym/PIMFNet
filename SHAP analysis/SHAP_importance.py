import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import time
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import gc

warnings.filterwarnings('ignore')


os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== 字体配置 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

print("=" * 80)
print("SHAP Feature Importance Analysis")
print("=" * 80)
print(f"\nTotal features: {len(all_feature_names)}")
print(f"  - TIGGE features: {len(tigge_feature_names)}")
print(f"  - DEM features: {len(dem_feature_names)}")
print(f"  - Interaction features: {len(interaction_feature_names)}")

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

class SHAPModelWrapper:


    def __init__(self, model, H=48, W=96, seq_len=4, device='cuda', batch_size=4):
        self.model = model
        self.H = H
        self.W = W
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size  
        self.time_features_t = torch.zeros(5).to(device)
        self.time_features_seq = torch.zeros(seq_len, 5).to(device)

    def predict(self, x):

        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x)
        else:
            x_tensor = x

        n_samples = x_tensor.shape[0]
        predictions = []

        for i in range(0, n_samples, self.batch_size):
            batch_x = x_tensor[i:i + self.batch_size].to(self.device)
            batch_size = batch_x.shape[0]

            # 分解特征
            tigge_features = batch_x[:, :10]
            dem_features = batch_x[:, 10:13]
            interaction_features = batch_x[:, 13:18]

            # 扩展到空间维度
            tigge_spatial = tigge_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.H, self.W)
            dem_spatial = dem_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.H, self.W)
            interaction_spatial = interaction_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.H, self.W)

            # 空间平均
            batch_pred = point_pred.mean(dim=(1, 2)).cpu().numpy()
            predictions.append(batch_pred)

            # 【关键】清理显存
            del tigge_spatial, dem_spatial, interaction_spatial, tigge_seq
            del time_features_t, time_features_seq, point_pred
            torch.cuda.empty_cache()

        return np.concatenate(predictions)


def main():
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting SHAP analysis...")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
    model = WindSpeedPredictor(H=48, W=96, tigge_features=10).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully!")

    n_samples = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Total samples in dataset: {n_samples}")

    # 提取特征数据
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Extracting features...")
    all_features = []

    for batch in test_loader:
        tigge = batch['tigge_spatial'].numpy()
        dem = batch['dem_spatial'].numpy()
        interaction = batch['interaction_spatial'].numpy()

        tigge_mean = tigge.mean(axis=(2, 3))
        dem_mean = dem.mean(axis=(2, 3))
        interaction_mean = interaction.mean(axis=(2, 3))

        features = np.concatenate([tigge_mean, dem_mean, interaction_mean], axis=1)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    print(f"Feature matrix shape: {all_features.shape}")
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating SHAP wrapper...")
    shap_model = SHAPModelWrapper(model, H=48, W=96, seq_len=4, device=device, batch_size=2)

    # 测试包装器
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing wrapper...")
    test_pred = shap_model.predict(all_features[:2])
    print(f"Test prediction shape: {test_pred.shape}")
    print(f"Test prediction values: {test_pred}")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Creating SHAP explainer...")
    print("Using KernelExplainer with reduced samples...")

    explainer = shap.KernelExplainer(shap_model.predict, background_data)
    print("✓ SHAP explainer created successfully!")

    test_samples_np = all_features[:100]

    try:
        shap_values = explainer.shap_values(test_samples_np, nsamples=50, silent=False)

        print(f"\nSHAP values shape: {shap_values.shape}")
        assert shap_values.shape == (100, 18), f"Expected shape (100, 18), got {shap_values.shape}"
        print("✓ SHAP values computed successfully!")

    except Exception as e:
        print(f"✗ Error computing SHAP values: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 保存SHAP值
    np.save(os.path.join(SAVE_DIR, 'shap_values.npy'), shap_values)
    np.save(os.path.join(SAVE_DIR, 'shap_test_samples.npy'), test_samples_np)
    print(f"✓ SHAP values saved to {SAVE_DIR}")

    # ==================== 可视化 ====================
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating visualizations...")

    try:
        # 1. SHAP Summary Plot
        print("Generating Summary Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, test_samples_np,
                          feature_names=all_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Summary plot saved")

        # 2. SHAP Bar Plot
        print("Generating Bar Plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_samples_np,
                          feature_names=all_feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Bar plot saved")

    except Exception as e:
        print(f"✗ Error generating SHAP plots: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nGenerating grouped bar plot...")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        ax1.bar(group_importance['Feature Group'], group_importance['Total Importance'], color=colors)
        ax1.set_ylabel('Total SHAP Importance', fontsize=14, fontweight='bold')
        ax1.set_title('(a) Total Feature Group Importance', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(axis='y', alpha=0.3)
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)

        ax2.bar(group_importance['Feature Group'], group_importance['Average Importance'], color=colors)
        ax2.set_ylabel('Average SHAP Importance', fontsize=14, fontweight='bold')
        ax2.set_title('(b) Average Feature Group Importance', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(axis='y', alpha=0.3)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'shap_group_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Grouped bar plot saved")

    except Exception as e:
        print(f"✗ Error generating grouped bar plot: {str(e)}")

   
    print("\nGenerating dependence plots for top 5 features...")
    try:
        top5_features = feature_importance.head(5)['Feature'].tolist()
        top5_indices = [all_feature_names.index(f) for f in top5_features]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (feat_name, feat_idx) in enumerate(zip(top5_features, top5_indices)):
            ax = axes[i]
            shap.dependence_plot(
                feat_idx,
                shap_values,
                test_samples_np,
                feature_names=all_feature_names,
                ax=ax,
                show=False
            )
            ax.set_title(f'({chr(97 + i)}) {feat_name}', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=11)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'shap_dependence_top5.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Dependence plots saved")

    except Exception as e:
        print(f"✗ Error generating dependence plots: {str(e)}")

    print("\nGenerating analysis report...")

    total_importance = tigge_importance + dem_importance + interaction_importance