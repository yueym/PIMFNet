
import pdb

# ==================== 环境准备 ====================
import torch  # 深度学习框架（GPU加速）
import glob  # 文件路径匹配（批量加载数据文件）
import cfgrib  # GRIB格式解码（TIGGE数据读取）
import matplotlib.pyplot as plt  # 数据可视化（特征分析绘图）
import numpy as np  # 数值计算核心库（矩阵运算）
import pandas as pd  # 时间序列处理（时间轴对齐）
import rioxarray as rxr  # 地理空间数据处理（DEM地形）
import xarray as xr  # 多维数据集处理（NetCDF/GRIB）
from scipy.interpolate import griddata  # 空间插值（双线性插值）
import os  # 系统路径操作

# ==================== 路径设置 ====================
raw_data_path = {
    "ERA5": "../data/ERA5/",
    "TIGGE": "../data/TIGGE/",
    "DEM": "../Data_Process/ASTGTM_003-20250307_080136/",
}
processed_path = "../processed_data/"

# ==================== 辅助函数 ====================

def calc_wind_speed(u, v):
    """计算风速（公式：√(u² + v²)）"""

    return np.sqrt(u ** 2 + v ** 2)


def plot_feature_importance(scores, features, filename):
    """绘制特征重要性柱状图"""

    plt.figure(figsize=(10, 6))
    plt.barh(features, scores)
    plt.title("特征重要性排序")

    plt.xlabel("重要性分数")

    plt.savefig(filename)
    plt.close()


# ==================== 0. 获取本机硬件信息 ====================
def get_hardware_info():
    # 获取本机信息，GPU/CPU
    print(torch.cuda.is_available())  # 应输出 True
    print(torch.cuda.current_device())  # 显示当前 GPU 索引
    print(torch.cuda.get_device_name())  # 显示 GPU 型号


# ==================== 1. TIGGE数据预处理（填补缺失月份） ====================
def load_grib_file(file):
    """加载GRIB文件中的所有变量，包括特殊处理问题变量"""
    print(f"加载文件: {file}")
    ds_list = []

    # 首先尝试加载大部分变量
    try:
        ds = xr.open_dataset(file, engine='cfgrib')
        ds_list.append(ds)
    except Exception as e:
        print(f"直接加载失败: {e}")

    # 特殊处理那些有问题的变量
    problem_vars = [
        {'shortName': 't2m', 'heightAboveGround': 2},
        {'shortName': 't2m', 'heightAboveGround': 10},  # 尝试不同的高度
        {'shortName': 'd2m', 'heightAboveGround': 2},
        {'shortName': 'd2m', 'heightAboveGround': 10},  # 尝试不同的高度
        {'shortName': 'mx2t6', 'step': 6},
        {'shortName': 'mn2t6', 'step': 6}
    ]

    for filter_keys in problem_vars:
        try:
            var_name = filter_keys['shortName']
            ds_var = xr.open_dataset(file, engine='cfgrib',
                                     backend_kwargs={'filter_by_keys': filter_keys})
            if ds_var.sizes:  # 检查是否为空
                print(f"成功单独加载变量: {var_name} (filter: {filter_keys})")
                ds_list.append(ds_var)
        except Exception as e:
            print(f"单独加载变量 {filter_keys} 失败: {e}")

    # 尝试使用更多的过滤器组合加载t2m和d2m
    t2m_loaded = any('t2m' in ds.data_vars for ds in ds_list)
    d2m_loaded = any('d2m' in ds.data_vars for ds in ds_list)

    if not t2m_loaded or not d2m_loaded:
        try:
            # 尝试使用paramId和typeOfLevel过滤
            t2m_filters = {'paramId': 167, 'typeOfLevel': 'heightAboveGround'}
            d2m_filters = {'paramId': 168, 'typeOfLevel': 'heightAboveGround'}

            for filter_keys, var_name in [(t2m_filters, 't2m'), (d2m_filters, 'd2m')]:
                if (var_name == 't2m' and not t2m_loaded) or (var_name == 'd2m' and not d2m_loaded):
                    try:
                        ds_var = xr.open_dataset(file, engine='cfgrib',
                                                 backend_kwargs={'filter_by_keys': filter_keys})
                        if ds_var.sizes and var_name in ds_var.data_vars:
                            print(f"成功使用paramId和typeOfLevel加载变量: {var_name}")
                            ds_list.append(ds_var)
                            if var_name == 't2m':
                                t2m_loaded = True
                            else:
                                d2m_loaded = True
                    except Exception as e:
                        print(f"使用paramId和typeOfLevel加载变量 {var_name} 失败: {e}")
        except Exception as e:
            print(f"尝试额外过滤器失败: {e}")

    # 尝试使用更宽松的过滤条件
    if not t2m_loaded or not d2m_loaded:
        try:
            # 只使用paramId过滤
            if not t2m_loaded:
                try:
                    ds_t2m = xr.open_dataset(file, engine='cfgrib',
                                             backend_kwargs={'filter_by_keys': {'paramId': 167}})
                    if ds_t2m.sizes and 't2m' in ds_t2m.data_vars:
                        print(f"成功仅使用paramId加载变量: t2m")
                        ds_list.append(ds_t2m)
                        t2m_loaded = True
                except Exception as e:
                    print(f"仅使用paramId加载t2m失败: {e}")

            if not d2m_loaded:
                try:
                    ds_d2m = xr.open_dataset(file, engine='cfgrib',
                                             backend_kwargs={'filter_by_keys': {'paramId': 168}})
                    if ds_d2m.sizes and 'd2m' in ds_d2m.data_vars:
                        print(f"成功仅使用paramId加载变量: d2m")
                        ds_list.append(ds_d2m)
                        d2m_loaded = True
                except Exception as e:
                    print(f"仅使用paramId加载d2m失败: {e}")
        except Exception as e:
            print(f"尝试仅使用paramId过滤失败: {e}")

    # 如果上述方法都失败，按paramId逐个加载其他变量
    if not ds_list:
        param_ids = [
            129, 130, 131, 132, 133, 134, 135, 136, 137, 151,
            152, 155, 157, 165, 166, 172, 173, 228,
            235, 236, 238, 260
        ]  # 排除了167, 168, 121, 122，因为已经尝试过了
        for param_id in param_ids:
            try:
                ds = xr.open_dataset(file, engine='cfgrib',
                                     backend_kwargs={'filter_by_keys': {'paramId': param_id}})
                if ds.sizes:  # 检查是否为空
                    ds_list.append(ds)
            except Exception as e:
                print(f"加载paramId={param_id}失败: {e}")

    if ds_list:
        # 合并所有加载的变量
        combined_ds = xr.merge(ds_list, compat='override')
        print(f"成功加载 {len(combined_ds.data_vars)} 个变量")
        return combined_ds
    else:
        print(f"无法加载任何变量: {file}")
        return xr.Dataset()


def preprocess_tigge_missing():
    print("预处理TIGGE数据，填补2019年缺失月份...")
    base_dir = r"E:\qixiang\Obs_PDF\data\TIGGE\2019"
    tigge_files = sorted(glob.glob(os.path.join(base_dir, "*.grib")))
    expected_files = [os.path.join(base_dir, f"2019.{month:02d}.grib") for month in range(1, 13)]

    print("找到的文件:", tigge_files)
    missing_files = [f for f in expected_files if f not in tigge_files]
    print(f"2019年缺失的文件: {missing_files}")

    if missing_files:
        for missing_file in missing_files:
            if "2019.02" in missing_file:
                prev_file = missing_file.replace("2019.02", "2019.01")
                next_file = missing_file.replace("2019.02", "2019.03")
                alt_file_2018 = r"E:\qixiang\Obs_PDF\data\TIGGE\2018\2018.02.grib"
                alt_file_2020 = r"E:\qixiang\Obs_PDF\data\TIGGE\2020\2020.02.grib"

                print(f"处理缺失文件: {missing_file}")
                feb_2019_ds = xr.Dataset()

                # 方法1：多年平均法
                if os.path.exists(alt_file_2018) and os.path.exists(alt_file_2020):
                    print("使用2018年和2020年2月数据的平均值...")
                    ds_2018 = load_grib_file(alt_file_2018)
                    ds_2020 = load_grib_file(alt_file_2020)
                    common_vars = list(set(ds_2018.data_vars).intersection(ds_2020.data_vars))

                    for var in common_vars:
                        try:
                            var_2018 = ds_2018[var]
                            var_2020 = ds_2020[var]
                            if var_2018.dims == var_2020.dims:
                                avg_var = (var_2018 + var_2020) / 2
                                if 'time' in avg_var.dims:
                                    times = avg_var.time.values
                                    new_times = [pd.Timestamp(t).replace(year=2019) for t in times]
                                    avg_var = avg_var.assign_coords(time=new_times)
                                feb_2019_ds[var] = avg_var
                                print(f"成功计算变量平均值: {var}")
                        except Exception as e:
                            print(f"计算变量 {var} 平均值失败: {e}")

                # 方法2：插值法
                if len(feb_2019_ds.data_vars) < 26 and os.path.exists(prev_file) and os.path.exists(next_file):
                    print("尝试使用插值法填补变量...")
                    ds_jan = load_grib_file(prev_file)
                    ds_mar = load_grib_file(next_file)
                    common_vars = list(set(ds_jan.data_vars).intersection(ds_mar.data_vars))
                    feb_times = [pd.Timestamp(year=2019, month=2, day=day.day, hour=hour)
                                 for day in pd.date_range('2019-02-01', '2019-02-28')
                                 for hour in [0, 6, 12, 18]]

                    for var in common_vars:
                        try:
                            var_jan = ds_jan[var]
                            var_mar = ds_mar[var]
                            if var_jan.dims == var_mar.dims:
                                combined = xr.concat([var_jan, var_mar], dim='time')
                                interp_var = combined.interp(time=feb_times, method='linear')
                                feb_2019_ds[var] = interp_var
                                print(f"成功插值变量: {var}")
                        except Exception as e:
                            print(f"插值变量 {var} 失败: {e}")

                # 保存结果
                if len(feb_2019_ds.data_vars) > 0:
                    output_nc = os.path.join(base_dir, "2019.02.nc")
                    feb_2019_ds.to_netcdf(output_nc)
                    print(f"已生成NetCDF文件: {output_nc}，包含 {len(feb_2019_ds.data_vars)} 个变量")
                    with open(os.path.join(base_dir, "2019.02.use_nc"), "w") as f:
                        f.write(f"使用NetCDF文件: {output_nc}")
                else:
                    print(f"无法生成变量: {missing_file}")
    else:
        print("2019年无缺失文件。")


# ==================== 2. ERA5数据处理 ====================
def process_era5(years=["2021", "2022", "2023", "2024"]):

    print(f"处理{', '.join(years)}年ERA5数据...")

    # 定义目标插值网格
    target_lat_min, target_lat_max = 35.13, 47.0
    target_lon_min, target_lon_max = 103.0, 126.88
    target_lon = np.arange(target_lon_min, target_lon_max + 0.01, 0.25)
    target_lat = np.arange(target_lat_min, target_lat_max + 0.01, 0.25)

    yearly_datasets = {}

    for year in years:
        print(f"\n===== 处理{year}年数据 =====")

        # 获取u和v分量文件，排除.idx文件
        u_files = sorted([f for f in glob.glob(raw_data_path["ERA5"] + f"{year}/anl_surf125.033_ugrd.{year}*") if not f.endswith('.idx')])
        v_files = sorted([f for f in glob.glob(raw_data_path["ERA5"] + f"{year}/anl_surf125.034_vgrd.{year}*") if not f.endswith('.idx')])

        print(f"找到u分量文件: {len(u_files)}个")
        print(f"找到v分量文件: {len(v_files)}个")

        if not u_files or not v_files:
            print(f"警告: 未找到{year}年ERA5数据文件，请检查路径")
            continue

        # 允许u和v文件数量不完全匹配，尝试对齐
        print("加载u分量数据...")
        u_datasets = []
        for file in u_files:
            try:
                ds_u = xr.open_dataset(file, engine='cfgrib')
                u_datasets.append(ds_u)
                print(f"u分量文件: {os.path.basename(file)}, 时间范围: {ds_u.time.min().values} 到 {ds_u.time.max().values}")
            except Exception as e:
                print(f"处理u分量文件 {file} 时出错: {str(e)}")

        print("加载v分量数据...")
        v_datasets = []
        for file in v_files:
            try:
                ds_v = xr.open_dataset(file, engine='cfgrib')
                v_datasets.append(ds_v)
                print(f"v分量文件: {os.path.basename(file)}, 时间范围: {ds_v.time.min().values} 到 {ds_v.time.max().values}")
            except Exception as e:
                print(f"处理v分量文件 {file} 时出错: {str(e)}")

        if not u_datasets or not v_datasets:
            print(f"无法加载{year}年数据，跳过")
            continue

        # 合并u和v分量数据
        print("合并u分量数据...")
        merged_u = xr.concat(u_datasets, dim='time')
        print("合并v分量数据...")
        merged_v = xr.concat(v_datasets, dim='time')

        # 对齐时间轴，保留所有时间点（以u或v为基准）
        print("对齐u和v分量的时间轴...")
        all_times = np.union1d(merged_u.time.values, merged_v.time.values)
        merged_u = merged_u.reindex(time=all_times, method='nearest', tolerance=np.timedelta64(3, 'h'))
        merged_v = merged_v.reindex(time=all_times, method='nearest', tolerance=np.timedelta64(3, 'h'))

        # 检查变量名
        u_var_name = next((var for var in merged_u.data_vars if 'ugrd' in var or 'u' in var.lower()), None)
        v_var_name = next((var for var in merged_v.data_vars if 'vgrd' in var or 'v' in var.lower()), None)

        if u_var_name is None or v_var_name is None:
            print(f"错误: 无法识别{year}年的u或v分量变量名")
            print(f"u数据集变量: {list(merged_u.data_vars)}")
            print(f"v数据集变量: {list(merged_v.data_vars)}")
            continue

        merged_u = merged_u.rename({u_var_name: 'u10'})
        merged_v = merged_v.rename({v_var_name: 'v10'})

        # 合并u和v分量
        ds_combined = xr.merge([merged_u['u10'], merged_v['v10']])

        # 处理坐标名称
        coord_mapping = {'latitude': 'lat', 'longitude': 'lon'}
        ds_combined = ds_combined.rename({k: v for k, v in coord_mapping.items() if k in ds_combined.coords})

        if 'lat' not in ds_combined.coords or 'lon' not in ds_combined.coords:
            print(f"错误: {year}年数据缺少经纬度坐标: {list(ds_combined.coords)}")
            continue

        # 检查纬度顺序
        lat_values = ds_combined.lat.values
        is_lat_descending = lat_values[0] > lat_values[-1]
        print(f"纬度坐标顺序: {'降序' if is_lat_descending else '升序'}")

        # 筛选目标小时
        target_hours = [0, 6, 12, 18]
        ds6 = ds_combined.sel(time=ds_combined.time.dt.hour.isin(target_hours))

        # 裁剪到目标区域
        print("裁剪到目标区域...")
        try:
            ds6_clipped = ds6.sel(
                lat=slice(target_lat_max, target_lat_min) if is_lat_descending else slice(target_lat_min, target_lat_max),
                lon=slice(target_lon_min, target_lon_max)
            )
            print(f"裁剪后经度范围: {ds6_clipped.lon.min().item():.2f} 到 {ds6_clipped.lon.max().item():.2f}")
            print(f"裁剪后纬度范围: {ds6_clipped.lat.min().item():.2f} 到 {ds6_clipped.lat.max().item():.2f}")
        except Exception as e:
            print(f"裁剪时出错: {str(e)}, 使用原始数据")
            ds6_clipped = ds6

        # 空间插值到0.25°网格
        print("执行空间插值到0.25°网格...")
        try:
            ds_interp = ds6_clipped.interp(lat=target_lat, lon=target_lon, method='linear', kwargs={"fill_value": "extrapolate"})
        except Exception as e:
            print(f"线性插值失败: {str(e)}, 尝试最近邻插值")
            ds_interp = ds6_clipped.interp(lat=target_lat, lon=target_lon, method='nearest')

        # 计算风速
        print("计算风速...")
        ds_interp["era5_wind_speed"] = calc_wind_speed(ds_interp.u10, ds_interp.v10)
        ds_final = ds_interp[["era5_wind_speed"]]

        # 移除多余维度
        unwanted_dims = ['expver', 'step', 'surface', 'heightAboveGround']
        for dim in unwanted_dims:
            if dim in ds_final.dims:
                ds_final = ds_final.squeeze(dim)

        # 检查单年缺失值
        missing = ds_final.era5_wind_speed.isnull().sum().item()
        total = ds_final.era5_wind_speed.size
        print(f"单年缺失值: {missing}/{total} ({missing / total * 100:.2f}%)")

        # 空间插值填充NaN
        if missing > 0:
            print("检测到NaN值，执行空间插值...")
            points = np.array([(i, j) for i in target_lon for j in target_lat]).reshape(-1, 2)
            for t in ds_final.time:
                values = ds_final.jra55_wind_speed.sel(time=t).values.flatten()
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 0:  # 确保有有效数据
                    points_valid = points[valid_mask]
                    values_valid = values[valid_mask]
                    interp_values = griddata(points_valid, values_valid, points, method='nearest')
                    ds_final.jra55_wind_speed.sel(time=t).values = interp_values.reshape(len(target_lat), len(target_lon))

        yearly_datasets[year] = ds_final
        print(f"成功处理{year}年数据")

    # 检查和合并多年数据
    print("\n===== 合并多年数据 =====")
    if not yearly_datasets:
        raise ValueError("没有成功处理任何年份的数据")

    merged_ds = xr.concat(list(yearly_datasets.values()), dim='time').sortby('time')

    # 检查缺失值并尝试时间和空间插值
    missing = merged_ds.era5_wind_speed.isnull().sum().item()
    total = merged_ds.era5_wind_speed.size
    print(f"合并后缺失值: {missing}/{total} ({missing / total * 100:.2f}%)")

    if missing > 0:
        print("正在进行时间维度线性插值...")
        merged_ds["era5_wind_speed"] = merged_ds.era5_wind_speed.interpolate_na(dim='time', method='linear')
        missing_after = merged_ds.era5_wind_speed.isnull().sum().item()
        print(f"时间插值后缺失值: {missing_after}/{total} ({missing_after / total * 100:.2f}%)")

        if missing_after > 0:
            print("进行空间插值填充剩余NaN...")
            points = np.array([(i, j) for i in target_lon for j in target_lat]).reshape(-1, 2)
            for t in merged_ds.time:
                values = merged_ds.era5_wind_speed.sel(time=t).values.flatten()
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) > 0:
                    points_valid = points[valid_mask]
                    values_valid = values[valid_mask]
                    interp_values = griddata(points_valid, values_valid, points, method='nearest')
                    merged_ds.era5_wind_speed.sel(time=t).values = interp_values.reshape(len(target_lat), len(target_lon))

    # 最终缺失值检查
    missing_final = merged_ds.era5_wind_speed.isnull().sum().item()
    print(f"最终缺失值: {missing_final}/{total} ({missing_final / total * 100:.2f}%)")

    # 保存结果
    os.makedirs(processed_path + "ERA5", exist_ok=True)
    year_range = f"{years[0]}_{years[-1]}"
    output_file = processed_path + f"ERA5/ERA5_processed_{year_range}.nc"
    print(f"正在保存处理结果到: {output_file}")
    merged_ds.to_netcdf(output_file)
    print(f"ERA5处理完成！保存至: {output_file}")

# ==================== 2. TIGGE数据处理 ====================
def process_tigge(years=["2021", "2022", "2023", "2024"]):

    print(f"处理{', '.join(years)}年TIGGE数据...")

    # 获取所有指定年份的文件路径
    tigge_files = []
    for year in years:
        year_files = sorted(
            glob.glob(raw_data_path["TIGGE"] + f"{year}/{year}.*.grib") +
            glob.glob(raw_data_path["TIGGE"] + f"{year}/{year}.*.nc")
        )
        tigge_files.extend(year_files)
        print(f"找到 {year} 年数据文件: {len(year_files)} 个")

    print(f"总共找到 {len(tigge_files)} 个TIGGE文件")

    if not tigge_files:
        raise FileNotFoundError(f"未找到TIGGE数据文件，请检查路径: {raw_data_path['TIGGE']}{years[0]}/")

    # 验证文件是否存在
    for file in tigge_files:
        if not os.path.exists(file):
            print(f"警告: 文件不存在: {file}")
        else:
            print(f"文件存在: {file}")

    # 初始化空数据集列表
    ds_list = []
    files_processed = 0
    files_failed = 0

    # 加载所有变量
    for file in tigge_files:
        print(f"\n正在加载文件 ({files_processed + files_failed + 1}/{len(tigge_files)}): {file}")
        try:
            if file.endswith('.grib'):
                datasets = cfgrib.open_datasets(file)
                for i, ds in enumerate(datasets):
                    print(f"子数据集 {i}:")
                    print(f"文件维度: {ds.dims}")
                    print(f"时间范围: {ds['time'].min().values} 到 {ds['time'].max().values}")
                    print(f"Step 值: {ds['step'].values}")
                    for var in ds.data_vars:
                        missing = ds[var].isnull().sum().item()
                        total = ds[var].size
                        print(f"{var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")
                    ds_list.append(ds)
            elif file.endswith('.nc'):
                ds = xr.open_dataset(file)
                print(f"文件维度: {ds.dims}")
                print(f"时间范围: {ds['time'].min().values} 到 {ds['time'].max().values}")
                for var in ds.data_vars:
                    missing = ds[var].isnull().sum().item()
                    total = ds[var].size
                    print(f"{var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")
                ds_list.append(ds)
            files_processed += 1
        except Exception as e:
            print(f"加载文件失败: {file}, 错误: {e}")
            files_failed += 1
            continue

    # 这行应在循环外部 - 所有文件处理完后才检查总数
    print(f"\n总计加载了 {len(ds_list)} 个子数据集，成功加载 {files_processed} 个文件，失败 {files_failed} 个文件")
    expected_datasets = len(years) * 12  # 假设每年每月至少1个子数据集
    if len(ds_list) < expected_datasets:
        print(f"警告: 加载的子数据集数量 ({len(ds_list)}) 少于预期 ({expected_datasets})，部分文件可能读取失败")
        if len(ds_list) == 0:
            raise ValueError("没有成功加载任何数据集，无法继续处理")

    # 合并逻辑应在循环外部
    print("\n合并所有加载的数据集...")
    # 按变量分组
    ds_by_var = {}
    for ds in ds_list:
        for var in ds.data_vars:
            if var not in ds_by_var:
                ds_by_var[var] = []
            ds_by_var[var].append(ds[var])

    # 设置全部时间段的时间轴（修改：改为2020-2023）
    start_date = f"{years[0]}-01-01 00:00"
    end_date = f"{years[-1]}-12-31 12:00"
    target_time = pd.date_range(start_date, end_date, freq="12h")  # 所有年份的时间轴
    print(f"设置目标时间轴: {start_date} 到 {end_date}, 共 {len(target_time)} 个时间点")

    # 按时间拼接每个变量
    merged_vars = {}
    for var, var_ds_list in ds_by_var.items():
        try:
            # 按时间维度拼接
            combined = xr.concat(var_ds_list, dim='time')
            # 确保时间轴唯一并排序
            combined = combined.sortby('time').reindex(time=target_time, method=None)
            merged_vars[var] = combined
            print(
                f"成功拼接变量 {var}, 时间范围: {combined['time'].min().values} 到 {combined['time'].max().values}")
            missing = combined.isnull().sum().item()
            total = combined.size
            print(f"拼接后 - {var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")
        except Exception as e:
            print(f"拼接变量 {var} 失败: {e}")

    # 创建合并后的数据集，使用 compat='override' 避免 valid_time 冲突
    tigge_ds = xr.merge(list(merged_vars.values()), compat='override')

    # 检查并添加必要的坐标（避免重复）
    if 'time' not in tigge_ds.coords:
        tigge_ds = tigge_ds.assign_coords({'time': target_time})
        print("添加了时间坐标")

    if 'step' not in tigge_ds.coords and len(ds_list) > 0 and 'step' in ds_list[0].coords:
        tigge_ds = tigge_ds.assign_coords({'step': ds_list[0]['step']})
        print("添加了step坐标")

    # 处理经纬度坐标 - 确保统一使用lat/lon
    if 'lat' not in tigge_ds.coords and 'latitude' in tigge_ds.coords:
        tigge_ds = tigge_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        print("将latitude/longitude重命名为lat/lon")
    elif 'lat' not in tigge_ds.coords and len(ds_list) > 0:
        # 检查ds_list中是否有latitude或lat
        if 'latitude' in ds_list[0].coords:
            tigge_ds = tigge_ds.assign_coords({
                'lat': ds_list[0]['latitude'],
                'lon': ds_list[0]['longitude']
            })
            print("从子数据集添加了lat/lon坐标（源自latitude/longitude）")
        elif 'lat' in ds_list[0].coords:
            tigge_ds = tigge_ds.assign_coords({
                'lat': ds_list[0]['lat'],
                'lon': ds_list[0]['lon']
            })
            print("从子数据集添加了lat/lon坐标")

    print(f"\n合并后数据集信息：")
    print(tigge_ds)

    # 检查并重命名变量
    if "10u" in tigge_ds and "10v" in tigge_ds:
        tigge_ds = tigge_ds.rename({"10u": "u10", "10v": "v10"})
        print("已将10u/10v重命名为u10/v10")
    elif "u10" not in tigge_ds or "v10" not in tigge_ds:
        print("警告: 未找到u10或v10变量，检查可用变量:")
        for var in tigge_ds.data_vars:
            print(f" - {var}")
        if '10u' in tigge_ds.data_vars or '10v' in tigge_ds.data_vars:
            print("有类似的变量(10u/10v)可能需要重命名")
        if len(tigge_ds.data_vars) == 0:
            raise KeyError("数据集不包含任何变量，无法继续处理")

    # 展平前检查缺失值
    print("\n展平前缺失值检查:")
    for var in tigge_ds.data_vars:
        missing = tigge_ds[var].isnull().sum().item()
        total = tigge_ds[var].size
        print(f"{var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")

    # 展平 time 和 step 维度
    print("\n展平 time 和 step 维度...")
    if 'step' in tigge_ds.dims:
        # 创建新的时间点列表
        step_hours = tigge_ds['step'].values / np.timedelta64(1, 'h')
        valid_times = []
        for t in tigge_ds['time'].values:
            for s in step_hours:
                valid_times.append(pd.Timestamp(t) + pd.Timedelta(hours=float(s)))

        # 创建新数据集
        ds_flat = xr.Dataset(coords={'time': valid_times, 'lat': tigge_ds['lat'], 'lon': tigge_ds['lon']})

        # 展平数据
        time_len = len(tigge_ds['time'])
        step_len = len(tigge_ds['step'])
        expected_time_len = time_len * step_len

        for var_name in tigge_ds.data_vars:
            var = tigge_ds[var_name]
            if 'time' in var.dims and 'step' in var.dims:
                # 维度顺序
                dims = list(var.dims)
                time_idx = dims.index('time')
                step_idx = dims.index('step')

                # 新维度
                new_dims = ['time', 'lat', 'lon']  # 强制指定新维度，确保移除 step

                # 重塑数据
                data = var.values
                if data.shape[time_idx] == time_len and data.shape[step_idx] == step_len:
                    # 将 time 和 step 合并为一个维度
                    new_shape = [expected_time_len, var.shape[dims.index('lat')], var.shape[dims.index('lon')]]
                    flat_data = data.reshape(new_shape)
                    ds_flat[var_name] = (new_dims, flat_data)
                else:
                    print(f"警告: 变量 {var_name} 的维度 {data.shape} 与预期不符，跳过展平")
            else:
                ds_flat[var_name] = var  # 不含 step 的变量直接复制

        print(f"展平后维度: {ds_flat.dims}")
        print(f"展平后时间范围: {ds_flat['time'].min().values} 到 {ds_flat['time'].max().values}")
        print(f"展平后时间点数量: {len(ds_flat['time'])}")

        for var in ds_flat.data_vars:
            missing = ds_flat[var].isnull().sum().item()
            total = ds_flat[var].size
            print(f"展平后 - {var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")
    else:
        ds_flat = tigge_ds
        print("无 step 维度，直接使用原始时间")

    # 对齐到目标时间轴 - 为2020-2023年创建6小时间隔的时间轴（修改）
    target_time_flat = pd.date_range(f"{years[0]}-01-01 00:00", f"{years[-1]}-12-31 18:00", freq="6h")
    print(f"\n目标时间点数量: {len(target_time_flat)}")
    print(f"目标时间范围: {target_time_flat[0]} 到 {target_time_flat[-1]}")

    ds_aligned = ds_flat.reindex(time=target_time_flat, method=None)
    print(f"对齐后维度: {ds_aligned.dims}")
    for var in ds_aligned.data_vars:
        missing = ds_aligned[var].isnull().sum().item()
        total = ds_aligned[var].size
        print(f"对齐后 - {var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")

    # 插值填补小范围缺失
    print("\n执行时间维度线性插值...")
    time_origin = ds_aligned['time'].values[0]
    time_numeric = (ds_aligned['time'].values - time_origin) / np.timedelta64(1, 'h')
    ds_numeric = ds_aligned.copy()
    ds_numeric['time'] = time_numeric
    ds_interpolated = ds_numeric.interpolate_na(dim="time", method="linear")
    ds_interpolated['time'] = time_origin + (ds_interpolated['time'] * np.timedelta64(1, 'h')).astype(
        'timedelta64[h]')

    print(f"插值后维度: {ds_interpolated.dims}")
    for var in ds_interpolated.data_vars:
        missing = ds_interpolated[var].isnull().sum().item()
        total = ds_interpolated[var].size
        print(f"插值后 - {var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")

    # 处理集合成员（如果存在）
    processed_ds = xr.Dataset()
    for var in ds_interpolated.data_vars:
        if 'number' in ds_interpolated[var].dims:
            processed_ds[f"{var}_mean"] = ds_interpolated[var].mean('number')
            processed_ds[f"{var}_std"] = ds_interpolated[var].std('number')
            processed_ds[f"{var}_max"] = ds_interpolated[var].max('number')
            processed_ds[f"{var}_min"] = ds_interpolated[var].min('number')
            processed_ds[f"{var}_median"] = ds_interpolated[var].median('number')
            print(f"处理了集合成员变量: {var} (计算了mean/std/max/min/median)")
        else:
            processed_ds[var] = ds_interpolated[var]

    # 空间插值到 0.25° 网格
    print("\n准备插值网格...")
    target_lon = np.arange(103, 126.88, 0.25)
    target_lat = np.arange(35.13, 47, 0.25)
    print(f"目标网格: 经度 {len(target_lon)} 点 ({target_lon[0]}-{target_lon[-1]})，纬度 {len(target_lat)} 点 ({target_lat[0]}-{target_lat[-1]})")

    grid_lon, grid_lat = np.meshgrid(target_lon, target_lat)
    source_lon = processed_ds.lon.values
    source_lat = processed_ds.lat.values
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    source_points = np.column_stack((source_lon_grid.ravel(), source_lat_grid.ravel()))
    print(f"源网格: 经度 {len(source_lon)} 点，纬度 {len(source_lat)} 点")

    interpolated_data = {}
    vars_processed = 0
    vars_failed = 0

    for var in processed_ds.data_vars:
        print(f"正在处理变量 ({vars_processed + vars_failed + 1}/{len(processed_ds.data_vars)}): {var}")
        try:
            if "lon" in processed_ds[var].dims and "lat" in processed_ds[var].dims:
                time_steps = len(processed_ds.time)
                interp_values = np.zeros((time_steps, len(target_lat), len(target_lon)))

                # 添加进度显示
                print(f"开始插值 {var}，共 {time_steps} 个时间步...")
                for t in range(time_steps):
                    if t % 100 == 0 or t == time_steps - 1:
                        print(f"  插值进度: {t+1}/{time_steps} ({(t+1)/time_steps*100:.1f}%)")

                    current_data = processed_ds[var].isel(time=t).values
                    interp = griddata(source_points, current_data.ravel(), (grid_lon, grid_lat), method='linear')

                    # 使用最近邻法填补线性插值后的缺失值
                    if np.any(np.isnan(interp)):
                        nearest_interp = griddata(source_points, current_data.ravel(),
                                                 (grid_lon, grid_lat), method='nearest')
                        # 只替换缺失值
                        mask = np.isnan(interp)
                        interp[mask] = nearest_interp[mask]

                    interp_values[t] = interp

                interpolated_data[var] = (["time", "lat", "lon"], interp_values)
                print(f"成功插值变量: {var}")
                vars_processed += 1
            else:
                interpolated_data[var] = processed_ds[var]
                print(f"变量 {var} 不需要空间插值，直接添加")
                vars_processed += 1
        except Exception as e:
            print(f"处理变量 {var} 时出错: {str(e)}")
            print(f"变量 {var} 的维度: {processed_ds[var].dims}, 形状: {processed_ds[var].shape}")
            vars_failed += 1
            continue

    print(f"\n空间插值完成，成功处理 {vars_processed} 个变量，失败 {vars_failed} 个变量")

    tigge_highres = xr.Dataset(
        interpolated_data,
        coords={"time": processed_ds.time, "lat": target_lat, "lon": target_lon}
    )

    # 计算风速
    if "u10" in tigge_highres and "v10" in tigge_highres:
        print("\n计算风速...")
        tigge_highres["tigge_wind_speed"] = calc_wind_speed(tigge_highres["u10"], tigge_highres["v10"])
        print("成功计算风速并添加为tigge_wind_speed变量")

        # 保存前检查完整性
        print("\n填补缺失值并检查完整性...")
        for var in tigge_highres.data_vars:
            if tigge_highres[var].isnull().any():
                print(f"变量 {var} 存在缺失值，执行时间维度线性插值填补...")
                tigge_highres[var] = tigge_highres[var].interpolate_na(dim='time', method='linear')

                # 检查是否还有缺失值，尝试最近邻插值
                if tigge_highres[var].isnull().any():
                    print(f"  线性插值后仍有缺失值，尝试最近邻插值...")
                    tigge_highres[var] = tigge_highres[var].interpolate_na(dim='time', method='nearest')

                    # 如果还有缺失值，使用变量均值填充
                    if tigge_highres[var].isnull().any():
                        mean_value = float(tigge_highres[var].mean(skipna=True).values)
                        print(f"  最近邻插值后仍有缺失值，使用变量均值{mean_value:.4f}填充...")
                        tigge_highres[var] = tigge_highres[var].fillna(mean_value)

            missing = tigge_highres[var].isnull().sum().item()
            total = tigge_highres[var].size
            print(f"保存前检查 - 变量 {var}: {missing}/{total} 缺失值 ({missing / total * 100:.2f}%)")

    # 保存结果 - 使用年份范围命名文件
    year_range = f"{years[0]}_{years[-1]}"
    output_path = processed_path + f"TIGGE/TIGGE_processed_{year_range}.nc"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在

    print(f"\n正在保存数据集到: {output_path}")
    print(f"数据集大小: {tigge_highres.nbytes / (1024**3):.2f} GB")
    print(f"数据集维度: {tigge_highres.dims}")
    print(f"数据集变量数量: {len(tigge_highres.data_vars)}")
    tigge_highres.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
    print(f"TIGGE处理完成！保存至: {output_path}，包含 {len(tigge_highres.data_vars)} 个变量")

# ==================== 3. DEM数据处理 ====================
def process_dem():

    print("处理DEM数据...")

    # 目标网格设置（与TIGGE保持一致）
    target_lon = np.arange(103, 126.88, 0.25)
    target_lat = np.arange(35.13, 47, 0.25)
    grid_lon, grid_lat = np.meshgrid(target_lon, target_lat)

    try:
        # 1. 加载DEM数据
        dem_files = sorted(glob.glob(raw_data_path["DEM"] + "*.tif"))
        if not dem_files:
            raise FileNotFoundError(f"未找到DEM数据文件，请检查路径: {raw_data_path['DEM']}")

        print(f"找到{len(dem_files)}个DEM文件")

        # 创建空数组存储拼接结果
        dem_data = np.zeros_like(grid_lon)
        valid_count = np.zeros_like(grid_lon)

        # 2. 逐个处理DEM文件
        for file in dem_files:
            try:
                # print(f"处理文件: {file}")
                # 读取数据
                ds = rxr.open_rasterio(file)

                # 获取经纬度范围
                lon_min, lon_max = ds.x.min().item(), ds.x.max().item()
                lat_min, lat_max = ds.y.min().item(), ds.y.max().item()

                # 判断是否在目标区域内
                if (lon_max < target_lon[0] or lon_min > target_lon[-1] or
                        lat_max < target_lat[0] or lat_min > target_lat[-1]):
                    print(f"文件 {file} 不在目标区域内，跳过")
                    continue

                # 投影到目标网格
                ds = ds.rio.reproject("EPSG:4326")  # 确保使用WGS84坐标系

                # 裁剪到目标区域
                ds = ds.sel(x=slice(target_lon[0], target_lon[-1]),
                            y=slice(target_lat[-1], target_lat[0]))

                # 插值到目标网格
                interp_dem = ds.interp(x=target_lon, y=target_lat, method="linear")

                # 更新结果（去除无效值）
                valid_mask = ~np.isnan(interp_dem.values[0])
                dem_data[valid_mask] += interp_dem.values[0][valid_mask]
                valid_count[valid_mask] += 1

            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
                continue

        # 计算平均值
        dem_data = np.where(valid_count > 0, dem_data / valid_count, np.nan)

        # 3. 计算地形特征
        print("计算地形特征...")

        # 计算网格间距（米）
        dx = np.diff(target_lon).mean() * 111000  # 经度1度约111km
        dy = np.diff(target_lat).mean() * 111000

        # 坡度（度）
        gradient_x = np.gradient(dem_data, dx, axis=1)
        gradient_y = np.gradient(dem_data, dy, axis=0)
        slope = np.arctan(np.sqrt(gradient_x ** 2 + gradient_y ** 2)) * 180 / np.pi

        # 坡向（度，北为0，顺时针）
        aspect = np.arctan2(-gradient_x, gradient_y) * 180 / np.pi + 180

        # 地形起伏度（局部高程标准差）
        relief = np.zeros_like(dem_data)
        window_size = 3  # 3x3窗口
        for i in range(window_size // 2, dem_data.shape[0] - window_size // 2):
            for j in range(window_size // 2, dem_data.shape[1] - window_size // 2):
                window = dem_data[i - window_size // 2:i + window_size // 2 + 1,
                         j - window_size // 2:j + window_size // 2 + 1]
                relief[i, j] = np.nanstd(window)

        # 4. 保存结果
        print("保存处理结果...")
        dem_ds = xr.Dataset(
            {
                "elevation": (["lat", "lon"], dem_data),
                "slope": (["lat", "lon"], slope),
                "aspect": (["lat", "lon"], aspect),
                "relief": (["lat", "lon"], relief)
            },
            coords={
                "lat": target_lat,
                "lon": target_lon
            }
        )

        # 添加变量属性
        dem_ds.elevation.attrs["units"] = "m"
        dem_ds.elevation.attrs["long_name"] = "海拔高度"
        dem_ds.slope.attrs["units"] = "degrees"
        dem_ds.slope.attrs["long_name"] = "坡度"
        dem_ds.aspect.attrs["units"] = "degrees"
        dem_ds.aspect.attrs["long_name"] = "坡向"
        dem_ds.relief.attrs["units"] = "m"
        dem_ds.relief.attrs["long_name"] = "地形起伏度"

        # 保存到文件
        dem_ds.to_netcdf(processed_path + "DEM/DEM_processed.nc")
        print("DEM处理完成！保存至:", processed_path + "DEM/DEM_processed.nc")

    except Exception as e:
        print(f"DEM处理失败: {str(e)}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 执行处理流程
    get_hardware_info()  # 获取硬件信息
    # preprocess_tigge_missing()  # 填补缺失的2020-2023年数据
    process_era5()
    # process_tigge()  # 处理TIGGE
    # process_dem()  # 处理DEM（保持不变）

    # 检查时间对齐
    era5_file = processed_path + "ERA5/ERA5_processed_2021_2024.nc"
    tigge_file = processed_path + "TIGGE/TIGGE_processed_2021_2024.nc"
    check_time_alignment(era5_file, tigge_file)

    print("所有数据处理完成！结果保存在:", processed_path)