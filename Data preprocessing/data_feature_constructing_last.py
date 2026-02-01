# -*- coding: utf-8 -*-
"""
任务：基于物理规律构建气象-地形交互特征
功能：
  1. 从已筛选的TIGGE和DEM数据集加载数据
  2. 基于大气物理方程构建5个核心交互特征
  3. 标准化所有特征到[0,1]范围
  4. 保存为训练集/验证集/测试集（NetCDF格式）
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import gc
import time
from datetime import datetime
from pathlib import Path

# ==================== 路径设置 ====================
# 输入路径（已筛选的数据集）
input_data_path = {
    "ERA5": r"E:\yym2\qixiang\Obs_PDF\processed_data_v2_features\ERA5\ERA5_final_2021_2024.nc",
    "TIGGE": r"E:\yym2\qixiang\Obs_PDF\processed_data_v2_features\TIGGE\TIGGE_final_2021_2024.nc",
    "DEM": r"E:\yym2\qixiang\Obs_PDF\processed_data_v2_features\DEM\DEM_final.nc"
}

# 输出路径
output_path = Path(r"E:\yym2\qixiang\Obs_PDF\final")
output_path.mkdir(parents=True, exist_ok=True)


# ==================== 辅助函数 ====================
def log_progress(message, start_time=None):
    """记录进度信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        elapsed = time.time() - start_time
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        print(f"[{timestamp}] {message} (耗时: {int(elapsed_min)}分{int(elapsed_sec)}秒)")
    else:
        print(f"[{timestamp}] {message}")
    return time.time()


def print_memory_usage(label=""):
    """打印当前内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    print(f"内存使用 {label}: {memory_gb:.2f} GB")


# ==================== 物理交互特征构建 ====================
def build_physics_based_interactions(tigge_df, dem_df):
    """
    基于大气物理方程构建5个核心交互特征

    参数:
        tigge_df: DataFrame, 包含TIGGE气象参数
        dem_df: DataFrame, 包含DEM地形参数

    返回:
        interactions_df: DataFrame, 包含5个物理交互特征
    """
    log_progress("开始构建物理交互特征...")

    interactions = {}

    # ========== 特征1: 风速-地形起伏交互 (最重要) ==========
    # 物理意义: 地形粗糙度对风速的阻力效应(wind speed,relief)
    # 公式: wind_terrain_drag = tigge_wind_speed × (1 + α × relief)
    # α为地形阻力系数，这里使用relief的归一化值作为权重
    log_progress("构建特征1: 风速-地形起伏交互 (wind_terrain_drag)")

    # 归一化relief到合理范围，避免极端值
    relief_normalized = (tigge_df['tigge_wind_speed'].values *
                         (1 + 0.1 * dem_df['relief'].values / dem_df['relief'].values.std()))
    interactions['wind_terrain_drag'] = relief_normalized
    print(f"   wind_terrain_drag: 均值={relief_normalized.mean():.4f}, 标准差={relief_normalized.std():.4f}")

    # ========== 特征2: 风矢量-坡向交互 (次重要) ==========
    # 物理意义: 风速矢量与地形朝向的耦合，决定风的遮挡和引导效应（u10,v10,slope)
    # 公式: wind_aspect_coupling = sqrt((u10×cos(aspect))² + (v10×sin(aspect))²)
    # aspect转换为弧度，计算风速分量在坡向上的投影
    log_progress("构建特征2: 风矢量-坡向交互 (wind_aspect_coupling)")

    aspect_rad = np.deg2rad(dem_df['aspect'].values)
    u10 = tigge_df['u10'].values
    v10 = tigge_df['v10'].values

    # 计算风速在坡向上的有效分量（考虑风向与坡向的夹角）
    wind_projection = np.sqrt(
        (u10 * np.cos(aspect_rad)) ** 2 +
        (v10 * np.sin(aspect_rad)) ** 2
    )
    interactions['wind_aspect_coupling'] = wind_projection
    print(f"   wind_aspect_coupling: 均值={wind_projection.mean():.4f}, 标准差={wind_projection.std():.4f}")

    # ========== 特征3: 温度-海拔垂直递减 ==========
    # 物理意义: 大气温度垂直递减率 (约6.5K/km)，反映局地热力条件(t2m,elevation)
    # 公式: thermal_lapse = t2m × exp(-γ × elevation / 1000)
    # γ为干绝热递减率系数 (约0.0065 K/m)
    log_progress("构建特征3: 温度-海拔垂直递减 (thermal_lapse)")

    gamma = 0.0065  # 干绝热递减率 (K/m)
    elevation_km = dem_df['elevation'].values / 1000.0  # 转换为km
    t2m = tigge_df['t2m'].values

    # 考虑海拔对温度的修正
    thermal_correction = t2m * np.exp(-gamma * elevation_km)
    interactions['thermal_lapse'] = thermal_correction
    print(f"   thermal_lapse: 均值={thermal_correction.mean():.4f}, 标准差={thermal_correction.std():.4f}")

    # ========== 特征4: 气压-海拔修正 ==========
    # 物理意义: 静力学方程的地形修正，气压随海拔的指数衰减(sp,elevation)
    # 公式: pressure_elevation = sp × exp(-elevation / H)
    # H为标高 (约8500m)
    log_progress("构建特征4: 气压-海拔修正 (pressure_elevation)")

    H = 8500.0  # 大气标高 (m)
    sp = tigge_df['sp'].values
    elevation = dem_df['elevation'].values

    # 气压的海拔修正（气压随海拔指数衰减）
    pressure_corrected = sp * np.exp(-elevation / H)
    interactions['pressure_elevation'] = pressure_corrected
    print(f"   pressure_elevation: 均值={pressure_corrected.mean():.4f}, 标准差={pressure_corrected.std():.4f}")

    # ========== 特征5: 辐射-坡向热力驱动 ==========
    # 物理意义: 坡向对太阳辐射的调制，驱动局地热力环流(ssr,slope)
    # 公式: radiation_aspect = ssr × cos(aspect - 180°)
    # 南坡（aspect≈180°）获得最大辐射，北坡最小
    log_progress("构建特征5: 辐射-坡向热力驱动 (radiation_aspect)")

    ssr = tigge_df['ssr'].values
    aspect_rad = np.deg2rad(dem_df['aspect'].values)

    # 计算坡向对辐射的增强/减弱系数
    # aspect=180°(南坡)时cos(0)=1最大，aspect=0°(北坡)时cos(180°)=-1最小
    aspect_factor = np.cos(aspect_rad - np.pi)  # 相对南向的辐射系数
    radiation_modulated = ssr * (1 + aspect_factor) / 2  # 归一化到[0, ssr]
    interactions['radiation_aspect'] = radiation_modulated
    print(f"   radiation_aspect: 均值={radiation_modulated.mean():.4f}, 标准差={radiation_modulated.std():.4f}")

    # 转换为DataFrame
    interactions_df = pd.DataFrame(interactions)

    log_progress(f"物理交互特征构建完成，共生成 {len(interactions)} 个特征")
    print(f"\n交互特征统计摘要:")
    print(interactions_df.describe())

    return interactions_df


# ==================== 数据加载和处理 ====================
def load_and_process_data():
    """加载数据并构建交互特征"""
    start_time = time.time()
    log_progress("=" * 100)
    log_progress("开始数据加载和处理流程")
    log_progress("=" * 100)
    print_memory_usage("处理前")

    # 1. 加载原始数据
    log_progress("步骤1: 加载ERA5、TIGGE、DEM数据...")
    era5 = xr.open_dataset(input_data_path["ERA5"])
    tigge = xr.open_dataset(input_data_path["TIGGE"])
    dem = xr.open_dataset(input_data_path["DEM"])

    log_progress(f"数据加载完成:")
    print(f"   ERA5时间范围: {era5.time.values.min()} 至 {era5.time.values.max()}")
    print(f"   TIGGE时间范围: {tigge.time.values.min()} 至 {tigge.time.values.max()}")
    print(f"   DEM空间范围: lat({dem.lat.values.min():.2f}, {dem.lat.values.max():.2f}), "
          f"lon({dem.lon.values.min():.2f}, {dem.lon.values.max():.2f})")

    # 2. 合并数据集
    log_progress("步骤2: 合并数据集...")
    merged = xr.merge([era5, tigge, dem], join='inner')
    print(f"   合并后数据形状: {dict(merged.dims)}")

    # 3. 按年份划分数据
    log_progress("步骤3: 按年份划分数据集...")
    merged['time'] = pd.to_datetime(merged.time.values)

    train_data = merged.sel(time=slice('2021-01-01', '2022-12-31'))
    val_data = merged.sel(time=slice('2023-01-01', '2023-12-31'))
    test_data = merged.sel(time=slice('2024-01-01', '2024-12-31'))

    print(f"   训练集(2021-2022): {len(train_data.time)} 时间点")
    print(f"   验证集(2023): {len(val_data.time)} 时间点")
    print(f"   测试集(2024): {len(test_data.time)} 时间点")

    # 4. 转换为DataFrame
    log_progress("步骤4: 转换为DataFrame格式...")
    train_df = train_data.to_dataframe().reset_index().dropna(subset=['era5_wind_speed'])
    val_df = val_data.to_dataframe().reset_index().dropna(subset=['era5_wind_speed'])
    test_df = test_data.to_dataframe().reset_index().dropna(subset=['era5_wind_speed'])

    print(f"   训练集样本数: {len(train_df)}")
    print(f"   验证集样本数: {len(val_df)}")
    print(f"   测试集样本数: {len(test_df)}")

    # 释放内存
    del merged, train_data, val_data, test_data, era5
    gc.collect()
    print_memory_usage("DataFrame转换后")

    log_progress("数据加载和处理完成", start_time)
    return train_df, val_df, test_df, tigge, dem


def extract_and_standardize_features(train_df, val_df, test_df, tigge, dem):
    """提取特征、构建交互特征并标准化"""
    start_time = time.time()
    log_progress("=" * 100)
    log_progress("开始特征提取和标准化")
    log_progress("=" * 100)

    # 1. 定义特征列表
    log_progress("步骤1: 定义特征列表...")

    # TIGGE气象特征（10个）
    tigge_features = [
        'tigge_wind_speed',  # TIGGE预报风速
        'orog',  # 地形高度
        'sm',  # 土壤湿度
        'lsm',  # 陆海掩膜
        'v10',  # 10米V风分量
        'ssr',  # 地表净短波辐射
        'u10',  # 10米U风分量
        'msl',  # 平均海平面气压
        't2m',  # 2米温度
        'sp'  # 地表气压
    ]

    # DEM地形特征（3个）
    dem_features = ['relief', 'elevation', 'aspect']

    # 时间特征
    time_features_list = ['hour', 'day', 'month', 'season']

    # 验证特征是否存在
    for feat in tigge_features:
        if feat not in train_df.columns:
            print(f"   警告: TIGGE特征 '{feat}' 不在数据集中")

    for feat in dem_features:
        if feat not in train_df.columns:
            print(f"   警告: DEM特征 '{feat}' 不在数据集中")

    print(f"   TIGGE特征: {len(tigge_features)}个")
    print(f"   DEM特征: {len(dem_features)}个")
    print(f"   时间特征: {len(time_features_list)}个")

    # 2. 构建物理交互特征
    log_progress("步骤2: 构建物理交互特征...")

    train_interactions = build_physics_based_interactions(
        train_df[tigge_features], train_df[dem_features]
    )
    val_interactions = build_physics_based_interactions(
        val_df[tigge_features], val_df[dem_features]
    )
    test_interactions = build_physics_based_interactions(
        test_df[tigge_features], test_df[dem_features]
    )

    interaction_features = list(train_interactions.columns)
    print(f"   生成交互特征: {len(interaction_features)}个")
    for i, feat in enumerate(interaction_features, 1):
        print(f"      {i}. {feat}")

    # 3. 标准化TIGGE特征
    log_progress("步骤3: 标准化TIGGE特征...")
    scaler_tigge = MinMaxScaler(feature_range=(0, 1))

    X_train_tigge = scaler_tigge.fit_transform(train_df[tigge_features].fillna(0))
    X_val_tigge = scaler_tigge.transform(val_df[tigge_features].fillna(0))
    X_test_tigge = scaler_tigge.transform(test_df[tigge_features].fillna(0))

    print(f"   TIGGE特征标准化完成: {X_train_tigge.shape}")

    # 4. 标准化DEM特征
    log_progress("步骤4: 标准化DEM特征...")
    scaler_dem = MinMaxScaler(feature_range=(0, 1))

    X_train_dem = scaler_dem.fit_transform(train_df[dem_features].fillna(0))
    X_val_dem = scaler_dem.transform(val_df[dem_features].fillna(0))
    X_test_dem = scaler_dem.transform(test_df[dem_features].fillna(0))

    print(f"   DEM特征标准化完成: {X_train_dem.shape}")

    # 5. 标准化交互特征
    log_progress("步骤5: 标准化交互特征...")
    scaler_interaction = MinMaxScaler(feature_range=(0, 1))

    X_train_interaction = scaler_interaction.fit_transform(train_interactions)
    X_val_interaction = scaler_interaction.transform(val_interactions)
    X_test_interaction = scaler_interaction.transform(test_interactions)

    print(f"   交互特征标准化完成: {X_train_interaction.shape}")

    # 6. 标准化目标变量
    log_progress("步骤6: 标准化目标变量...")
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    y_train = scaler_y.fit_transform(train_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(val_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(test_df['era5_wind_speed'].values.reshape(-1, 1)).flatten()

    print(f"   目标变量标准化完成: {y_train.shape}")

    # 7. 处理时间特征
    log_progress("步骤7: 处理时间特征...")
    time_features_dict = {}

    for phase, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        df['time'] = pd.to_datetime(df['time'])
        time_cols = pd.DataFrame({
            'year': df['time'].dt.year,
            'month': df['time'].dt.month,
            'day': df['time'].dt.day,
            'hour': df['time'].dt.hour,
            'season': df['time'].dt.month % 12 // 3 + 1
        })
        time_features_dict[phase] = time_cols.values.astype(np.int64)

    print(f"   时间特征提取完成")

    # 8. 保存标准化器
    log_progress("步骤8: 保存标准化器...")
    joblib.dump(scaler_tigge, output_path / 'tigge_feature_scaler.pkl')
    joblib.dump(scaler_dem, output_path / 'dem_feature_scaler.pkl')
    joblib.dump(scaler_interaction, output_path / 'interaction_feature_scaler.pkl')
    joblib.dump(scaler_y, output_path / 'target_scaler.pkl')

    print(f"   标准化器已保存至: {output_path}")

    # 释放内存
    del train_df, val_df, test_df
    gc.collect()
    print_memory_usage("特征提取后")

    log_progress("特征提取和标准化完成", start_time)

    return (X_train_tigge, X_val_tigge, X_test_tigge,
            X_train_dem, X_val_dem, X_test_dem,
            X_train_interaction, X_val_interaction, X_test_interaction,
            y_train, y_val, y_test,
            time_features_dict,
            tigge_features, dem_features, interaction_features)


def save_final_datasets(X_train_tigge, X_val_tigge, X_test_tigge,
                        X_train_dem, X_val_dem, X_test_dem,
                        X_train_interaction, X_val_interaction, X_test_interaction,
                        y_train, y_val, y_test,
                        time_features_dict,
                        tigge_features, dem_features, interaction_features):
    """保存最终数据集为NetCDF格式"""
    start_time = time.time()
    log_progress("=" * 100)
    log_progress("开始保存最终数据集")
    log_progress("=" * 100)

    # 数据维度验证
    log_progress("步骤1: 验证数据维度...")
    print(f"   TIGGE特征: Train{X_train_tigge.shape}, Val{X_val_tigge.shape}, Test{X_test_tigge.shape}")
    print(f"   DEM特征: Train{X_train_dem.shape}, Val{X_val_dem.shape}, Test{X_test_dem.shape}")
    print(
        f"   交互特征: Train{X_train_interaction.shape}, Val{X_val_interaction.shape}, Test{X_test_interaction.shape}")
    print(f"   目标变量: Train{y_train.shape}, Val{y_val.shape}, Test{y_test.shape}")

    time_feature_labels = ['year', 'month', 'day', 'hour', 'season']

    # 创建训练集
    log_progress("步骤2: 创建训练集数据集...")
    train_ds = xr.Dataset(
        data_vars={
            "tigge_features": (["sample", "tigge_feature"], X_train_tigge),
            "dem_features": (["sample", "dem_feature"], X_train_dem),
            "interaction_features": (["sample", "interaction_feature"], X_train_interaction),
            "target": (["sample"], y_train),
            "time_features": (["sample", "time_feature"], time_features_dict['train'])
        },
        coords={
            "sample": np.arange(X_train_tigge.shape[0]),
            "tigge_feature": tigge_features,
            "dem_feature": dem_features,
            "interaction_feature": interaction_features,
            "time_feature": time_feature_labels
        },
        attrs={
            "description": "训练集数据（2021-2022）",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tigge_features_count": len(tigge_features),
            "dem_features_count": len(dem_features),
            "interaction_features_count": len(interaction_features),
            "total_samples": X_train_tigge.shape[0]
        }
    )

    # 创建验证集
    log_progress("步骤3: 创建验证集数据集...")
    val_ds = xr.Dataset(
        data_vars={
            "tigge_features": (["sample", "tigge_feature"], X_val_tigge),
            "dem_features": (["sample", "dem_feature"], X_val_dem),
            "interaction_features": (["sample", "interaction_feature"], X_val_interaction),
            "target": (["sample"], y_val),
            "time_features": (["sample", "time_feature"], time_features_dict['val'])
        },
        coords={
            "sample": np.arange(X_val_tigge.shape[0]),
            "tigge_feature": tigge_features,
            "dem_feature": dem_features,
            "interaction_feature": interaction_features,
            "time_feature": time_feature_labels
        },
        attrs={
            "description": "验证集数据（2023）",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tigge_features_count": len(tigge_features),
            "dem_features_count": len(dem_features),
            "interaction_features_count": len(interaction_features),
            "total_samples": X_val_tigge.shape[0]
        }
    )

    # 创建测试集
    log_progress("步骤4: 创建测试集数据集...")
    test_ds = xr.Dataset(
        data_vars={
            "tigge_features": (["sample", "tigge_feature"], X_test_tigge),
            "dem_features": (["sample", "dem_feature"], X_test_dem),
            "interaction_features": (["sample", "interaction_feature"], X_test_interaction),
            "target": (["sample"], y_test),
            "time_features": (["sample", "time_feature"], time_features_dict['test'])
        },
        coords={
            "sample": np.arange(X_test_tigge.shape[0]),
            "tigge_feature": tigge_features,
            "dem_feature": dem_features,
            "interaction_feature": interaction_features,
            "time_feature": time_feature_labels
        },
        attrs={
            "description": "测试集数据（2024）",
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tigge_features_count": len(tigge_features),
            "dem_features_count": len(dem_features),
            "interaction_features_count": len(interaction_features),
            "total_samples": X_test_tigge.shape[0]
        }
    )

    # 保存数据集
    try:
        log_progress("步骤5: 保存数据集到磁盘...")

        train_file = output_path / "train.nc"
        val_file = output_path / "val.nc"
        test_file = output_path / "test.nc"

        train_ds.to_netcdf(train_file)
        print(f"   ✓ 训练集已保存: {train_file}")

        val_ds.to_netcdf(val_file)
        print(f"   ✓ 验证集已保存: {val_file}")

        test_ds.to_netcdf(test_file)
        print(f"   ✓ 测试集已保存: {test_file}")

        # 输出文件大小
        train_size = train_file.stat().st_size / (1024 ** 2)
        val_size = val_file.stat().st_size / (1024 ** 2)
        test_size = test_file.stat().st_size / (1024 ** 2)

        print(f"\n   文件大小:")
        print(f"      train.nc: {train_size:.2f} MB")
        print(f"      val.nc: {val_size:.2f} MB")
        print(f"      test.nc: {test_size:.2f} MB")
        print(f"      总计: {train_size + val_size + test_size:.2f} MB")

    except Exception as e:
        print(f"   ✗ 保存数据集时出错: {e}")
        raise

    log_progress("数据集保存完成", start_time)

    return train_ds, val_ds, test_ds


def verify_final_datasets():
    """验证保存的数据集"""
    log_progress("=" * 100)
    log_progress("验证保存的数据集")
    log_progress("=" * 100)

    datasets = {
        "训练集": output_path / "train.nc",
        "验证集": output_path / "val.nc",
        "测试集": output_path / "test.nc"
    }

    for name, path in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"正在验证: {name}")
        print(f"{'=' * 80}")

        if not path.exists():
            print(f"   ✗ 文件不存在: {path}")
            continue

        ds = xr.open_dataset(path)

        print(f"\n📁 文件信息:")
        print(f"   路径: {path}")
        print(f"   大小: {path.stat().st_size / (1024 ** 2):.2f} MB")

        print(f"\n📐 维度:")
        for dim, size in ds.dims.items():
            print(f"   {dim:<25}: {size:,}")

        print(f"\n📦 数据变量:")
        for var in ds.data_vars:
            print(f"   {var:<25}: {ds[var].dims} - {ds[var].shape}")

        print(f"\n🏷️  坐标变量:")
        for coord in ds.coords:
            if coord not in ds.dims:
                coord_data = ds.coords[coord]
                if coord_data.size <= 10:
                    print(f"   {coord:<25}: {coord_data.values.tolist()}")
                else:
                    print(f"   {coord:<25}: (共 {coord_data.size} 个值)")

        print(f"\n📊 属性信息:")
        for attr, value in ds.attrs.items():
            print(f"   {attr}: {value}")

        # 数据统计
        print(f"\n📈 数据统计:")
        print(f"   目标变量 (target):")
        target_data = ds['target'].values
        print(f"      最小值: {target_data.min():.6f}")
        print(f"      最大值: {target_data.max():.6f}")
        print(f"      平均值: {target_data.mean():.6f}")
        print(f"      标准差: {target_data.std():.6f}")

        ds.close()

    log_progress("数据集验证完成")


# ==================== 主流程 ====================
def main():
    """主流程函数"""
    total_start_time = time.time()

    print("\n" + "=" * 100)
    print("基于物理规律的气象-地形交互特征构建系统".center(100))
    print("=" * 100)
    log_progress(f"开始执行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 步骤1: 加载和处理数据
        log_progress("\n【阶段1/4】数据加载和预处理")
        train_df, val_df, test_df, tigge, dem = load_and_process_data()

        # 步骤2: 提取特征、构建交互特征并标准化
        log_progress("\n【阶段2/4】特征提取和标准化")
        (X_train_tigge, X_val_tigge, X_test_tigge,
         X_train_dem, X_val_dem, X_test_dem,
         X_train_interaction, X_val_interaction, X_test_interaction,
         y_train, y_val, y_test,
         time_features_dict,
         tigge_features, dem_features, interaction_features) = extract_and_standardize_features(
            train_df, val_df, test_df, tigge, dem
        )

        # 步骤3: 保存最终数据集
        log_progress("\n【阶段3/4】保存最终数据集")
        train_ds, val_ds, test_ds = save_final_datasets(
            X_train_tigge, X_val_tigge, X_test_tigge,
            X_train_dem, X_val_dem, X_test_dem,
            X_train_interaction, X_val_interaction, X_test_interaction,
            y_train, y_val, y_test,
            time_features_dict,
            tigge_features, dem_features, interaction_features
        )

        # 步骤4: 验证数据集
        log_progress("\n【阶段4/4】验证保存的数据集")
        verify_final_datasets()

        # 最终总结
        print("\n" + "=" * 100)
        print("处理完成！".center(100))
        print("=" * 100)

        print("\n📊 最终数据集结构:")
        print(f"\n训练集 (train.nc):")
        print(train_ds)

        print(f"\n验证集 (val.nc):")
        print(val_ds)

        print(f"\n测试集 (test.nc):")
        print(test_ds)

        print("\n✅ 特征构成:")
        print(f"   1. TIGGE气象特征 ({len(tigge_features)}个):")
        for i, feat in enumerate(tigge_features, 1):
            print(f"      {i:2d}. {feat}")

        print(f"\n   2. DEM地形特征 ({len(dem_features)}个):")
        for i, feat in enumerate(dem_features, 1):
            print(f"      {i}. {feat}")

        print(f"\n   3. 物理交互特征 ({len(interaction_features)}个):")
        for i, feat in enumerate(interaction_features, 1):
            print(f"      {i}. {feat}")
            # 添加物理意义说明
            if feat == 'wind_terrain_drag':
                print(f"         → 物理意义: 地形粗糙度对风速的阻力效应")
            elif feat == 'wind_aspect_coupling':
                print(f"         → 物理意义: 风速矢量与地形朝向的耦合")
            elif feat == 'thermal_lapse':
                print(f"         → 物理意义: 温度随海拔的垂直递减")
            elif feat == 'pressure_elevation':
                print(f"         → 物理意义: 气压随海拔的指数衰减")
            elif feat == 'radiation_aspect':
                print(f"         → 物理意义: 坡向对太阳辐射的调制")

        print(f"\n   4. 时间特征 (5个):")
        print(f"      1. year")
        print(f"      2. month")
        print(f"      3. day")
        print(f"      4. hour")
        print(f"      5. season")

        print(f"\n   总特征数: {len(tigge_features) + len(dem_features) + len(interaction_features) + 5}")

        print("\n📁 输出文件位置:")
        print(f"   数据集目录: {output_path}")
        print(f"   - train.nc (训练集)")
        print(f"   - val.nc (验证集)")
        print(f"   - test.nc (测试集)")
        print(f"   - tigge_feature_scaler.pkl (TIGGE标准化器)")
        print(f"   - dem_feature_scaler.pkl (DEM标准化器)")
        print(f"   - interaction_feature_scaler.pkl (交互特征标准化器)")
        print(f"   - target_scaler.pkl (目标变量标准化器)")

        print("\n🎯 下一步操作建议:")
        print("   1. 使用标准化器对新数据进行预处理")
        print("   2. 构建深度学习模型（如Transformer、LSTM等）")
        print("   3. 训练模型时可分别使用三类特征或组合使用")
        print("   4. 评估物理交互特征对预测性能的贡献")

        print("\n⏱️  总处理时间:")
        total_time = time.time() - total_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"   {hours}小时 {minutes}分钟 {seconds}秒")

        print("\n" + "=" * 100)
        print_memory_usage("处理完成后")

    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()