import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import optuna
from optuna.samplers import TPESampler
import time
import os
import gc
from torch.cuda.amp import autocast, GradScaler
import warnings

warnings.filterwarnings('ignore')

# ✅ 修改1：导入第二个模型的类和函数
from train_task2 import (
    WindSpeedPredictor,
    WindDataset,
    MultiTaskLoss,
    compute_crps,
    compute_interval_metrics
)


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 贝叶斯优化目标函数 ====================
def objective(trial):
    """
    Optuna目标函数（修复版）
    """

    # ==================== 1. 超参数搜索空间 ====================
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 24, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.35)
    ltc_hidden_dim = trial.suggest_categorical("ltc_hidden_dim", [192, 200, 208, 216, 224, 232, 240, 248, 256])
    cbam_reduction = trial.suggest_categorical("cbam_reduction", [8, 12, 16, 20, 24])
    n_epochs = trial.suggest_int("n_epochs", 6, 12)
    accumulation_steps = trial.suggest_categorical("accumulation_steps", [1, 2, 4])

    # 多任务损失权重
    alpha = trial.suggest_float("alpha", 0.5, 2.0, step=0.1)
    beta = trial.suggest_float("beta", 0.01, 0.3, step=0.01)
    gamma = trial.suggest_float("gamma", 0.01, 0.3, step=0.01)

    try:
        # ==================== 2. 数据加载 ====================
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial.number} - Loading datasets")

        H, W = 48, 96
        train_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\train.nc", H, W)
        val_ds = WindDataset(r"E:\yym2\qixiang\Obs_PDF\final\val.nc", H, W)

        # ✅ 修复错误2：Windows兼容的num_workers
        num_workers = 0 if os.name == 'nt' else 4

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Train: {len(train_ds)}, Val: {len(val_ds)}")

        # ==================== 3. 设备配置 ====================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")

        # ==================== 4. 模型初始化 ====================
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing model")

        model = WindSpeedPredictor(
            H=H, W=W,
            tigge_features=10,
            dropout_rate=dropout_rate,
            ltc_hidden_dim=ltc_hidden_dim,
            cbam_reduction=cbam_reduction
        ).to(device)

        # LTC权重初始化
        for m in model.ltc.ode_func.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in model.ltc.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # ==================== 5. 训练设置 ====================
        criterion = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
        scaler = GradScaler()

        # ==================== 6. 训练循环 ====================
        best_val_mae = float('inf')

        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial.number}, Epoch {epoch + 1}/{n_epochs}")

            # ==================== 6.1 训练阶段 ====================
            model.train()
            train_loss = 0.0
            train_count = 0

            train_point_preds = []
            train_mus = []
            train_sigmas = []
            train_interval_preds = []
            train_targets = []

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
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

                    scaler.scale(loss / accumulation_steps).backward()

                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    train_point_preds.append(point_pred.detach().cpu().float())
                    train_mus.append(mu.detach().cpu().float())
                    train_sigmas.append(sigma.detach().cpu().float())
                    train_interval_preds.append(interval_pred.detach().cpu().float())
                    train_targets.append(target.detach().cpu().float())

                    train_loss += loss.item() * tigge_spatial.size(0)
                    train_count += tigge_spatial.size(0)

                    # ✅✅✅ 新增:显示loss详细分解 ✅✅✅
                    if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                        progress = (batch_idx + 1) / len(train_loader) * 100
                        avg_loss_so_far = train_loss / train_count if train_count > 0 else 0

                        # 提取loss分量
                        point_loss = loss_dict.get('point_loss', 0.0)
                        crps_loss = loss_dict.get('crps_loss', 0.0)
                        interval_loss = loss_dict.get('interval_loss', 0.0)

                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"Trial {trial.number}, Epoch {epoch + 1}/{n_epochs}, "
                              f"Batch {batch_idx + 1}/{len(train_loader)} ({progress:.1f}%) | "
                              f"Loss: {loss.item():.6f} (Avg: {avg_loss_so_far:.6f}) | "
                              f"Point: {point_loss:.6f}, CRPS: {crps_loss:.6f}, Interval: {interval_loss:.6f}")

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[WARNING] OOM at batch {batch_idx}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e

            avg_train_loss = train_loss / train_count if train_count > 0 else float('inf')

            # ==================== 计算训练指标 ====================
            train_point_preds_tensor = torch.cat(train_point_preds)
            train_mus_tensor = torch.cat(train_mus)
            train_sigmas_tensor = torch.cat(train_sigmas)
            train_interval_preds_tensor = torch.cat(train_interval_preds)
            train_targets_tensor = torch.cat(train_targets)

            train_point_preds_flat = train_point_preds_tensor.numpy().flatten()
            train_mus_flat = train_mus_tensor.numpy().flatten()
            train_sigmas_flat = train_sigmas_tensor.numpy().flatten()
            train_targets_flat = train_targets_tensor.numpy().flatten()

            # 点预测指标
            train_mae = np.mean(np.abs(train_point_preds_flat - train_targets_flat))
            train_rmse = np.sqrt(np.mean((train_point_preds_flat - train_targets_flat) ** 2))
            train_mean = np.mean(train_targets_flat)
            ss_tot = np.sum((train_targets_flat - train_mean) ** 2)
            ss_res = np.sum((train_targets_flat - train_point_preds_flat) ** 2)
            train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # CRPS指标
            train_crps = compute_crps(
                torch.from_numpy(train_mus_flat),
                torch.from_numpy(train_sigmas_flat),
                torch.from_numpy(train_targets_flat)
            ).item()

            # 区间预测指标
            train_interval_metrics = compute_interval_metrics(
                train_interval_preds_tensor,
                train_targets_tensor,
                train_point_preds_tensor
            )

            # ✅ 修改：添加MWP95输出
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training Summary - "
                  f"Loss: {avg_train_loss:.6f} | "
                  f"MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f} | "
                  f"CRPS: {train_crps:.4f} | "
                  f"CP95: {train_interval_metrics['CP_95']:.4f}, MWP95: {train_interval_metrics['MWP_95']:.4f}, MC95: {train_interval_metrics['MC_95']:.4f}")

            # ==================== 6.2 验证阶段 ====================
            model.eval()

            val_point_preds = []
            val_mus = []
            val_sigmas = []
            val_interval_preds = []
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

                        val_point_preds.append(point_pred.cpu().float())
                        val_mus.append(mu.cpu().float())
                        val_sigmas.append(sigma.cpu().float())
                        val_interval_preds.append(interval_pred.cpu().float())
                        val_targets.append(target.cpu().float())

                        if batch_idx % 50 == 0 or batch_idx == len(val_loader) - 1:
                            progress = (batch_idx + 1) / len(val_loader) * 100
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                  f"Validation Batch {batch_idx + 1}/{len(val_loader)} ({progress:.1f}%)")

                    except Exception as e:
                        continue

            # ==================== 计算验证指标 ====================
            val_point_preds_tensor = torch.cat(val_point_preds)
            val_mus_tensor = torch.cat(val_mus)
            val_sigmas_tensor = torch.cat(val_sigmas)
            val_interval_preds_tensor = torch.cat(val_interval_preds)
            val_targets_tensor = torch.cat(val_targets)

            val_point_preds_flat = val_point_preds_tensor.numpy().flatten()
            val_mus_flat = val_mus_tensor.numpy().flatten()
            val_sigmas_flat = val_sigmas_tensor.numpy().flatten()
            val_targets_flat = val_targets_tensor.numpy().flatten()

            val_mae = np.mean(np.abs(val_point_preds_flat - val_targets_flat))
            val_rmse = np.sqrt(np.mean((val_point_preds_flat - val_targets_flat) ** 2))
            val_mean = np.mean(val_targets_flat)
            ss_tot = np.sum((val_targets_flat - val_mean) ** 2)
            ss_res = np.sum((val_targets_flat - val_point_preds_flat) ** 2)
            val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            val_crps = compute_crps(
                torch.from_numpy(val_mus_flat),
                torch.from_numpy(val_sigmas_flat),
                torch.from_numpy(val_targets_flat)
            ).item()

            val_interval_metrics = compute_interval_metrics(
                val_interval_preds_tensor,
                val_targets_tensor,
                val_point_preds_tensor
            )

            # ✅ 修改：添加MWP95输出
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation Summary - "
                  f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f} | "
                  f"CRPS: {val_crps:.4f} | "
                  f"CP95: {val_interval_metrics['CP_95']:.4f}, MWP95: {val_interval_metrics['MWP_95']:.4f}, MC95: {val_interval_metrics['MC_95']:.4f}")

            epoch_duration = time.time() - epoch_start_time
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Epoch {epoch + 1}/{n_epochs} completed in {epoch_duration / 60:.2f} minutes\n")

            # 学习率调整
            scheduler.step()

            # Early stopping
            if val_mae < best_val_mae:
                best_val_mae = val_mae

            # Optuna中间值报告
            trial.report(val_mae, epoch)
            if trial.should_prune():
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

        # ==================== 7. 计算最终综合得分 ====================
        # ✅ 异常值检测（修改：检查MWP而非MC）
        if np.isnan(val_mae) or np.isnan(val_crps) or np.isnan(val_interval_metrics['MWP_95']):
            print(f"[WARNING] Trial {trial.number} has NaN metrics")
            return 10.0

        # R²裁剪到合理范围
        val_r2 = max(min(val_r2, 1.0), -1.0)

        # ==================== 7.1 点预测得分（30%）====================
        mae_score = val_mae / 0.05  # 期望MAE < 0.05
        r2_score = max(val_r2, 0)  # 负R²视为0
        point_score = 0.5 * mae_score + 0.5 * (1 - r2_score)

        # ==================== 7.2 概率预测得分（35%）====================
        crps_score = val_crps / 0.04  # 期望CRPS < 0.04

        # ==================== 7.3 区间预测得分（35%）====================
        # ✅ 修改：使用 MWP 替代 MC
        mwp_score = val_interval_metrics['MWP_95'] / 0.30  # 期望MWP < 30%
        cp_deviation = abs(val_interval_metrics['CP_95'] - 0.95) / 0.05  # 期望CP ≈ 0.95

        interval_score = 0.50 * mwp_score + 0.50 * cp_deviation

        # ==================== 7.4 最终综合得分 ====================
        combined_score = 0.35 * point_score + 0.30 * crps_score + 0.35 * interval_score

        # ==================== 7.5 详细输出（新增）====================
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Trial {trial.number} Score Breakdown:")
        print(f"  Point Score:    {point_score:.4f} (35%) = 0.5*MAE_score({mae_score:.4f}) + 0.5*(1-R²_score({r2_score:.4f}))")
        print(f"  CRPS Score:     {crps_score:.4f} (30%)")
        print(f"  Interval Score: {interval_score:.4f} (35%) = 0.5*MWP_score({mwp_score:.4f}) + 0.5*CP_dev({cp_deviation:.4f})")
        print(f"  ────────────────────────────────────")
        print(f"  Combined Score: {combined_score:.6f} (minimize)")
        print(f"  ════════════════════════════════════\n")

        # ✅ 修改：保存详细指标（添加分数组成）
        try:
            trial.set_user_attr("val_mae", float(val_mae))
            trial.set_user_attr("val_rmse", float(val_rmse))
            trial.set_user_attr("val_r2", float(val_r2))
            trial.set_user_attr("val_crps", float(val_crps))
            trial.set_user_attr("val_cp_95", float(val_interval_metrics['CP_95']))
            trial.set_user_attr("val_mwp_95", float(val_interval_metrics['MWP_95']))
            trial.set_user_attr("val_mc_95", float(val_interval_metrics['MC_95']))

            # ✅ 新增：保存各部分得分
            trial.set_user_attr("point_score", float(point_score))
            trial.set_user_attr("crps_score", float(crps_score))
            trial.set_user_attr("interval_score", float(interval_score))
            trial.set_user_attr("mwp_score", float(mwp_score))
            trial.set_user_attr("cp_deviation", float(cp_deviation))
        except Exception as e:
            print(f"[WARNING] Failed to set user attributes: {str(e)}")

        return combined_score

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Critical error in trial {trial.number}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 10.0

    finally:
        try:
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


# ==================== 主函数 ====================
def run_optimization(n_trials=75, study_name="wind_speed_multi_task_optimization"):
    """运行贝叶斯优化"""
    set_seed(42)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Bayesian Optimization for Multi-Task Model")
    print(f"  Study name: {study_name}")
    print(f"  Number of trials: {n_trials}")

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    study.enqueue_trial({
        "batch_size": 16,
        "learning_rate": 0.0003931296040961602,
        "weight_decay": 0.0001025927948297095,
        "dropout_rate": 0.24733185479083603,
        "ltc_hidden_dim": 216,
        "cbam_reduction": 16,
        "n_epochs": 8,
        "accumulation_steps": 1,
        "alpha": 1.0,
        "beta": 0.05,
        "gamma": 0.05
    })

    study.optimize(objective, n_trials=n_trials)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Composite Score): {best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # ✅ 修改：输出详细指标
    print(f"  Metrics:")
    for key in ["val_mae", "val_rmse", "val_r2", "val_crps", "val_cp_95", "val_mwp_95", "val_mc_95"]:
        if key in best_trial.user_attrs:
            print(f"    {key}: {best_trial.user_attrs[key]:.4f}")

    # ✅ 修改：输出得分组成
    print(f"  Score Components:")
    for key in ["point_score", "crps_score", "interval_score", "mwp_score", "cp_deviation"]:
        if key in best_trial.user_attrs:
            print(f"    {key}: {best_trial.user_attrs[key]:.4f}")

    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv(f"{study_name}_results.csv", index=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Results saved to {study_name}_results.csv")

    return study


if __name__ == "__main__":
    study = run_optimization(n_trials=75, study_name="wind_speed_multi_task_optimization_v4")