#!/usr/bin/env python3
"""
Label Permutation Test and Sanity Check Module
Integrated version of complete_sanity_check.py for src/ directory
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
from pathlib import Path
import sys
import os

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_target_column(dataset_name, df):
    """获取目标列"""
    target_mapping = {
        'biotoxin': 'VALUE',
        'cast': 'Bottom_D',
        'era5_daily': 'wind10',
        'cleaned_data': 'G2chla',
        'rolling_mean': 'G2chla', 
        'processed_seq': 'G2chla',
        'hydrographic': 'G2chla',
        'phyto_wide': 'Pseudo-nitzschia americana/brasiliana (cells l-1)',
        'phyto_long': 'GYMNODINIALES Karlodinium-like'
    }
    
    target_col = target_mapping.get(dataset_name)
    
    # 如果映射的列不存在，尝试其他可能的列
    if target_col not in df.columns:
        possible_targets = ['G2chla', 'chla', 'target', 'y', 'VALUE']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                break
        
        # 如果还是没找到，使用最后一个数值列
        if target_col not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
    
    return target_col

def prepare_features(dataset_name, df, target_col):
    """准备特征，移除可能导致泄漏的列"""
    exclude_cols = ['Date', 'date', 'time', 'Time', target_col]
    
    # 特殊处理某些数据集
    if dataset_name == 'cast':
        exclude_cols.extend(['Latitude', 'Longitude'])  # 地理坐标可能泄漏位置信息
    elif dataset_name == 'hydrographic':
        exclude_cols.extend(['LATITUDE', 'LONGITUDE'])
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 移除缺失值
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y

def quick_sanity_check(X, y, dataset_name, n_permutations=3):
    """快速sanity check"""
    print(f"🔬 {dataset_name}: {X.shape[0]}样本, {X.shape[1]}特征")
    
    # 1. 原始标签训练（简化版，只用一次分割）
    split_point = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练XGBoost
    model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    original_r2 = r2_score(y_test, y_pred)
    
    # 2. 置换标签训练
    permuted_r2_scores = []
    
    for perm in range(n_permutations):
        # 随机置换目标变量
        y_permuted = np.random.permutation(y.values)
        y_train_perm = y_permuted[:split_point]
        y_test_perm = y_permuted[split_point:]
        
        # 训练模型
        model_perm = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model_perm.fit(X_train_scaled, y_train_perm)
        y_pred_perm = model_perm.predict(X_test_scaled)
        
        r2_perm = r2_score(y_test_perm, y_pred_perm)
        permuted_r2_scores.append(r2_perm)
    
    # 3. 计算平均置换R²
    avg_permuted_r2 = np.mean(permuted_r2_scores)
    
    # 4. 判断是否通过sanity check
    pass_check = abs(avg_permuted_r2) < 0.15
    
    print(f"   原始R²: {original_r2:.4f}")
    print(f"   置换R²: {avg_permuted_r2:.4f}")
    print(f"   通过检验: {'✅' if pass_check else '❌'}")
    
    return {
        'dataset': dataset_name,
        'original_r2': original_r2,
        'permuted_r2': avg_permuted_r2,
        'pass_sanity_check': pass_check,
        'n_features': X.shape[1],
        'n_samples': X.shape[0]
    }

def run_all_datasets_sanity_check():
    """运行所有数据集的sanity check"""
    print("🔍 运行所有数据集的Sanity Check")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    
    all_datasets = [
        'biotoxin', 'cast', 'era5_daily', 'cleaned_data',
        'rolling_mean', 'processed_seq', 'hydrographic',
        'phyto_wide', 'phyto_long'
    ]
    
    all_results = []
    
    for dataset in all_datasets:
        print(f"\n📊 检查 {dataset}")
        
        try:
            # 加载数据
            data_path = base_path.parent / f'data/processed/{dataset}/clean.csv'
            df = pd.read_csv(data_path)
            
            # 获取目标列
            target_col = get_target_column(dataset, df)
            if target_col is None or target_col not in df.columns:
                print(f"❌ 无法找到目标列")
                continue
            
            # 准备特征
            X, y = prepare_features(dataset, df, target_col)
            
            if len(X) < 50:  # 样本太少
                print(f"❌ 样本数量不足: {len(X)}")
                all_results.append({
                    'dataset': dataset,
                    'original_r2': np.nan,
                    'permuted_r2': np.nan,
                    'pass_sanity_check': False,
                    'n_features': X.shape[1] if len(X) > 0 else 0,
                    'n_samples': len(X),
                    'error': 'Insufficient samples'
                })
                continue
            
            # 执行sanity check
            result = quick_sanity_check(X, y, dataset)
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ {dataset} 处理失败: {e}")
            all_results.append({
                'dataset': dataset,
                'original_r2': np.nan,
                'permuted_r2': np.nan,
                'pass_sanity_check': False,
                'n_features': 0,
                'n_samples': 0,
                'error': str(e)
            })
    
    return all_results

def generate_sanity_check_report(results):
    """生成sanity check报告"""
    print(f"\n📋 Sanity Check 总结报告")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    
    # 统计结果
    total_datasets = len(results_df)
    passed_datasets = results_df['pass_sanity_check'].sum()
    failed_datasets = total_datasets - passed_datasets
    
    print(f"总数据集: {total_datasets}")
    print(f"通过检验: {passed_datasets}")
    print(f"未通过检验: {failed_datasets}")
    
    # 显示详细结果
    print(f"\n详细结果:")
    for _, row in results_df.iterrows():
        status = "✅" if row['pass_sanity_check'] else "❌"
        original_r2 = row['original_r2'] if not pd.isna(row['original_r2']) else 'N/A'
        permuted_r2 = row['permuted_r2'] if not pd.isna(row['permuted_r2']) else 'N/A'
        print(f"  {status} {row['dataset']:15s}: 原始={original_r2}, 置换={permuted_r2}")
    
    # 未通过检验的数据集
    failed_df = results_df[~results_df['pass_sanity_check']]
    if not failed_df.empty:
        print(f"\n⚠️ 未通过检验的数据集:")
        for _, row in failed_df.iterrows():
            reason = row.get('error', '可能存在数据泄漏')
            print(f"   - {row['dataset']}: {reason}")
    
    # 保存结果
    base_path = Path(__file__).parent.parent.parent
    output_path = base_path / 'outputs' / 'tables' / 'complete_sanity_check_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 完整结果已保存: {output_path}")
    
    return results_df

def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    
    print("🚀 开始完整的Sanity Check验证")
    print("=" * 60)
    
    # 运行所有数据集检查
    all_results = run_all_datasets_sanity_check()
    
    # 生成报告
    report_df = generate_sanity_check_report(all_results)
    
    # 关键结论
    print(f"\n🎯 关键结论:")
    passed_count = report_df['pass_sanity_check'].sum()
    total_count = len(report_df)
    
    if passed_count >= total_count * 0.8:  # 80%通过率
        print(f"   ✅ {passed_count}/{total_count} 数据集通过sanity check")
        print(f"   ✅ 大部分结果是合理的，无明显数据泄漏")
        print(f"   ✅ 可以安全地在论文中报告这些结果")
    else:
        failed_datasets = report_df[~report_df['pass_sanity_check']]['dataset'].tolist()
        print(f"   ⚠️ {len(failed_datasets)} 个数据集未通过检验: {failed_datasets}")
        print(f"   ⚠️ 需要进一步调查和修复")
    
    return report_df

if __name__ == "__main__":
    main()
