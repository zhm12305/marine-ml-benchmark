import argparse, joblib, optuna, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils_io import read_cfg

ROOT = Path(__file__).resolve().parents[1]

# ---------------- model registry ----------------
def get_model(name, params):
    if name == "rf":
        return RandomForestRegressor(random_state=0, **params)
    if name == "xgb":
        return XGBRegressor(random_state=0, **params)
    if name == "svr":
        return SVR(**params)
    raise ValueError(name)

search_space = {
    "rf":  {"n_estimators": (50, 200), "max_depth": (5, 15)},  # 减少范围加快训练
    "xgb": {"learning_rate": (0.05, 0.2), "n_estimators": (100, 300),
            "max_depth": (3, 8)},
    "svr": {"C": (0.1, 10), "gamma": (1e-3, 1e-1)}
}

# ------------------------- main ------------------
def objective(trial, X, y, model_name):
    params = {}
    for k, rng in search_space[model_name].items():
        if k in ['n_estimators', 'max_depth']:
            params[k] = trial.suggest_int(k, *rng)
        else:
            params[k] = trial.suggest_float(k, *rng, log=True)
    model  = make_pipeline(StandardScaler(), get_model(model_name, params))
    cv     = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr, te in cv.split(X):
        model.fit(X[tr], y[tr])
        scores.append(model.score(X[te], y[te]))
    return -np.mean(scores)

def train_dataset(ds_key, model_name, cfg):
    print(f"🤖 训练 {ds_key} - {model_name}")

    # 检查数据文件
    data_path = ROOT / "data_proc" / ds_key / "clean.csv"
    if not data_path.exists():
        print(f"   ❌ 数据文件不存在")
        return None

    df = pd.read_csv(data_path)
    tcol = cfg["datasets"][ds_key].get("target_col")
    if not tcol:
        print(f"   ❌ 无目标列")
        return None

    if tcol not in df.columns:
        # 尝试查找相似的目标列
        possible_cols = [col for col in df.columns if any(keyword in col.lower()
                        for keyword in ['chla', 'chlorophyll', 'target', 'value', 'abundance'])]
        if possible_cols:
            tcol = possible_cols[0]
            print(f"   ⚠️  使用 {tcol} 替代原目标列")
        else:
            print(f"   ❌ 目标列 {tcol} 不存在")
            return None

    X = df.drop(columns=[tcol]).select_dtypes(include=[np.number]).to_numpy()
    y = df[tcol].to_numpy()

    if len(X) == 0 or len(y) == 0:
        print(f"   ❌ 数据为空")
        return None

    print(f"   📊 数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"   🔍 开始超参数搜索 ({cfg['global']['n_trials']} 次试验)...")

    study = optuna.create_study(
        direction='maximize',  # 最大化R²
        sampler=optuna.samplers.TPESampler(seed=cfg["global"]["random_seed"])
    )

    # 修改objective函数返回正值（R²）
    def modified_objective(trial):
        return -objective(trial, X, y, model_name)  # 转换为正值

    study.optimize(modified_objective, n_trials=cfg["global"]["n_trials"],
                   show_progress_bar=True)

    print(f"   ✅ 最佳 R²: {study.best_value:.4f}")

    best_model = make_pipeline(StandardScaler(),
                               get_model(model_name, study.best_params))
    best_model.fit(X, y)
    out_dir = ROOT / "models" / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / f"{model_name}.pkl")

    return {"dataset": ds_key, "model": model_name,
            "best_params": study.best_params, "score": study.best_value}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--model",  default="all")
    args = parser.parse_args()

    cfg = read_cfg()
    dsets = cfg["datasets"].keys() if args.dataset == "all" else [args.dataset]
    models = ["rf", "xgb", "svr"] if args.model == "all" else [args.model]

    print(f"🚀 开始训练 {len(dsets)} 个数据集 × {len(models)} 个模型")
    print("=" * 60)

    records = []
    total_combinations = sum(1 for d in dsets if cfg["datasets"][d].get("target_col")) * len(models)
    current = 0

    for d in dsets:
        if not cfg["datasets"][d].get("target_col"):
            print(f"⏭️  跳过 {d}: 无目标列")
            continue

        for m in models:
            current += 1
            print(f"\n[{current}/{total_combinations}] 当前组合: {d} + {m}")
            print("-" * 40)

            try:
                rec = train_dataset(d, m, cfg)
                if rec:
                    records.append(rec)
                    print(f"   ✅ 完成: R² = {rec['score']:.4f}")
                else:
                    print(f"   ❌ 失败: 返回空结果")
            except KeyboardInterrupt:
                print(f"\n⚠️  训练被用户中断")
                break
            except Exception as e:
                print(f"   ❌ 失败: {e}")
                continue
        else:
            continue  # 只有在内层循环正常完成时才继续
        break  # 如果内层循环被break，外层也break

    # 保存结果
    if records:
        results_df = pd.DataFrame(records)
        results_df.to_csv(ROOT / "tables" / "train_log.csv", index=False)
        print(f"\n🎉 训练完成!")
        print(f"✅ 成功训练: {len(records)} 个模型")
        print(f"📁 结果保存至: tables/train_log.csv")
        print(f"📁 模型保存至: models/")

        # 显示最佳结果
        if len(records) > 0:
            best_result = max(records, key=lambda x: x['score'])
            print(f"🏆 最佳模型: {best_result['dataset']} + {best_result['model']} (R² = {best_result['score']:.4f})")
    else:
        print(f"\n❌ 没有成功训练的模型")
