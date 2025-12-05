import argparse, joblib, optuna, numpy as np, pandas as pd
import json, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.utils_io import read_cfg
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]

# 设置设备（GPU优先）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用设备: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

def calculate_metrics(y_true, y_pred):
    """计算多个评估指标"""
    metrics = {}

    # R² Score
    metrics['R2'] = r2_score(y_true, y_pred)

    # Mean Absolute Error
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)

    # Root Mean Square Error
    metrics['RMSE'] = mean_squared_error(y_true, y_pred, squared=False)

    # Mean Absolute Percentage Error (处理除零问题)
    try:
        # 避免除零，只在目标值不为0时计算MAPE
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics['MAPE'] = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
        else:
            metrics['MAPE'] = float('inf')  # 所有真实值都为0
    except:
        metrics['MAPE'] = float('inf')

    # 相关系数
    try:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['Correlation'] = correlation if not np.isnan(correlation) else 0.0
    except:
        metrics['Correlation'] = 0.0

    # 检查指标有效性
    for key, value in metrics.items():
        if np.isnan(value) or np.isinf(value):
            if key == 'MAPE':
                metrics[key] = 999.0  # MAPE的默认大值
            else:
                metrics[key] = 0.0 if key in ['R2', 'Correlation'] else 999.0

    return metrics

# ================ 深度学习模型定义 ================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()

        # 降维处理：先将高维特征降到合理范围
        self.feature_reduction = nn.Linear(input_size, min(32, input_size//2))
        self.input_projection = nn.Linear(min(32, input_size//2), d_model)

        # 简化位置编码
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model) * 0.1)

        # 使用更小的Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 特征降维
        x = self.feature_reduction(x)
        x = torch.relu(x)

        # 输入投影
        x = self.input_projection(x)

        # 位置编码 - 确保在正确设备上
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc

        # Transformer编码
        x = self.transformer(x)

        # 全局平均池化 + 层归一化
        x = self.layer_norm(x.mean(dim=1))
        x = self.dropout(x)

        # 输出
        x = self.fc(x)
        return x

# ================ 模型注册表 ================
def get_model(name, params, input_size=None):
    if name == "rf":
        return RandomForestRegressor(random_state=42, **params)
    elif name == "xgb":
        return XGBRegressor(random_state=42, **params)
    elif name == "svr":
        return SVR(**params)
    elif name == "lstm":
        return LSTMModel(input_size=input_size, **params)
    elif name == "transformer":
        return TransformerModel(input_size=input_size, **params)
    else:
        raise ValueError(f"Unknown model: {name}")

def get_search_space(model_name, cfg):
    """从配置文件获取搜索空间"""
    if model_name not in cfg["global"]["models"]:
        raise ValueError(f"Model {model_name} not configured")

    # 将列表转换为元组
    search_space = cfg["global"]["models"][model_name]
    return {param: tuple(range_list) for param, range_list in search_space.items()}

# ================ 训练函数 ================
def train_sklearn_model(X, y, model_name, cfg):
    """训练sklearn模型"""
    search_space = get_search_space(model_name, cfg)
    
    def objective(trial):
        params = {}
        for param, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param] = trial.suggest_int(param, low, high)
            elif param in ['learning_rate', 'gamma']:
                # 对数尺度参数
                params[param] = trial.suggest_float(param, low, high, log=True)
            else:
                # 线性尺度参数
                params[param] = trial.suggest_float(param, low, high, log=False)
        
        model = make_pipeline(StandardScaler(), get_model(model_name, params))
        cv = TimeSeriesSplit(n_splits=cfg["global"]["cv_folds"])
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(r2_score(y[val_idx], pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(seed=cfg["global"]["random_seed"]))
    study.optimize(objective, n_trials=cfg["global"]["n_trials"], show_progress_bar=False)
    
    # 用最佳参数训练最终模型并计算多个指标
    best_model = make_pipeline(StandardScaler(), get_model(model_name, study.best_params))

    # 使用训练-测试分割来评估最终模型
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg["global"]["random_seed"])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # 计算多个评估指标
    metrics = calculate_metrics(y_test, y_pred)

    # 在全部数据上重新训练用于保存
    best_model.fit(X, y)

    return best_model, study.best_params, study.best_value, metrics

def train_deep_model(X, y, model_name, cfg):
    """训练深度学习模型"""
    # 数据分割
    train_ratio = cfg["global"]["train_ratio"]
    val_ratio = cfg["global"]["val_ratio"]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=1-train_ratio-val_ratio, random_state=cfg["global"]["random_seed"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio/(train_ratio+val_ratio), 
        random_state=cfg["global"]["random_seed"]
    )
    
    # 数据标准化
    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    
    # 转换为PyTorch张量并移动到GPU
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    
    search_space = get_search_space(model_name, cfg)
    
    def objective(trial):
        try:
            params = {}
            for param, (low, high) in search_space.items():
                if param == "learning_rate":
                    params[param] = trial.suggest_float(param, low, high, log=True)
                elif isinstance(low, int):
                    params[param] = trial.suggest_int(param, low, high)
                else:
                    params[param] = trial.suggest_float(param, low, high, log=False)

            lr = params.pop("learning_rate", 1e-3)
            model = get_model(model_name, params, input_size=X_train_scaled.shape[-1]).to(device)

            # 添加权重衰减和学习率调度
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=False
            )
        
            # 改进的训练循环
            model.train()
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(50):  # 减少epoch数以加快速度
                # 训练步骤
                optimizer.zero_grad()
                outputs = model(X_train_tensor).squeeze()
                train_loss = criterion(outputs, y_train_tensor)

                # 检查损失是否为NaN
                if torch.isnan(train_loss):
                    return 0.0  # 返回最差分数

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                train_loss.backward()
                optimizer.step()

                # 验证（每5个epoch）
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor).squeeze()
                        val_loss = criterion(val_outputs, y_val_tensor)

                        # 检查验证损失
                        if torch.isnan(val_loss):
                            return 0.0

                        # 早停检查
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            break

                        # 学习率调度
                        scheduler.step(val_loss)

                    model.train()

            # 最终验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_pred = scaler_y.inverse_transform(val_outputs.cpu().numpy().reshape(-1, 1)).flatten()
                val_true = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
                score = r2_score(val_true, val_pred)

                # 确保返回有效分数
                if np.isnan(score) or np.isinf(score):
                    return 0.0

                return max(0.0, score)  # 确保分数非负

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0  # 返回最差分数
    
    study = optuna.create_study(direction='maximize',
                               sampler=optuna.samplers.TPESampler(seed=cfg["global"]["random_seed"]))
    study.optimize(objective, n_trials=cfg["global"]["n_trials"]//2, show_progress_bar=False)  # 减少试验次数
    
    # 训练最终模型
    best_params = study.best_params.copy()
    lr = best_params.pop("learning_rate", 1e-3)
    final_model = get_model(model_name, best_params, input_size=X_train_scaled.shape[-1]).to(device)
    
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 合并训练和验证集进行最终训练
    X_final = np.concatenate([X_train_scaled, X_val_scaled])
    y_final = np.concatenate([y_train_scaled, y_val_scaled])
    X_final_tensor = torch.FloatTensor(X_final).to(device)
    y_final_tensor = torch.FloatTensor(y_final).to(device)
    
    final_model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = final_model(X_final_tensor).squeeze()
        loss = criterion(outputs, y_final_tensor)
        loss.backward()
        optimizer.step()
    
    # 包装模型以便保存
    class DeepModelWrapper:
        def __init__(self, model, scaler_X, scaler_y):
            self.model = model
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
            
        def predict(self, X):
            self.model.eval()
            X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                outputs = self.model(X_tensor).squeeze()
                predictions = self.scaler_y.inverse_transform(outputs.numpy().reshape(-1, 1)).flatten()
            return predictions
    
    wrapped_model = DeepModelWrapper(final_model, scaler_X, scaler_y)

    # 计算最终模型的多个评估指标
    final_model.eval()
    with torch.no_grad():
        # 使用验证集评估
        val_outputs = final_model(X_val_tensor).squeeze()
        val_pred = scaler_y.inverse_transform(val_outputs.cpu().numpy().reshape(-1, 1)).flatten()
        val_true = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

        # 计算多个指标
        metrics = calculate_metrics(val_true, val_pred)

    return wrapped_model, study.best_params, study.best_value, metrics

def train_dataset(ds_key, model_name, cfg):
    """训练单个数据集的模型"""
    print(f"训练 {ds_key} - {model_name}")
    
    # 检查是否有目标列
    if not cfg["datasets"][ds_key].get("target_col"):
        print(f"  跳过 {ds_key}: 无目标列")
        return None
    
    # 加载数据
    data_path = ROOT / "data_proc" / ds_key / "clean.csv"
    if not data_path.exists():
        print(f"  跳过 {ds_key}: 数据文件不存在")
        return None
    
    df = pd.read_csv(data_path)
    target_col = cfg["datasets"][ds_key]["target_col"]
    
    if model_name in ["lstm", "transformer"]:
        # 深度学习模型使用序列数据
        seq_path = ROOT / "data_proc" / ds_key / "sequences.npz"
        if not seq_path.exists():
            print(f"  跳过 {ds_key}: 序列数据不存在")
            return None
        
        data = np.load(seq_path)
        X, y = data['X'], data['y']
        
        if len(X) == 0:
            print(f"  跳过 {ds_key}: 序列数据为空")
            return None
        
        model, best_params, best_score, metrics = train_deep_model(X, y, model_name, cfg)
    else:
        # 传统机器学习模型
        if target_col not in df.columns:
            print(f"  跳过 {ds_key}: 目标列 {target_col} 不存在")
            return None

        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
        y = df[target_col].values

        if len(X) == 0 or len(y) == 0:
            print(f"  跳过 {ds_key}: 数据为空")
            return None

        model, best_params, best_score, metrics = train_sklearn_model(X, y, model_name, cfg)
    
    # 保存模型
    out_dir = ROOT / "models" / ds_key
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if model_name in ["lstm", "transformer"]:
        # 深度学习模型需要特殊保存方式
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'scaler_X': model.scaler_X,
            'scaler_y': model.scaler_y,
            'model_params': best_params
        }, out_dir / f"{model_name}.pth")
    else:
        joblib.dump(model, out_dir / f"{model_name}.pkl")
    
    # 保存最佳参数
    with open(out_dir / f"{model_name}_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"  完成 {ds_key} - {model_name}: R² = {best_score:.4f}")
    
    # 构建结果字典，包含多个评估指标
    result = {
        "dataset": ds_key,
        "model": model_name,
        "best_params": best_params,
        "best_score": best_score,
        "data_shape": X.shape if 'X' in locals() else "N/A"
    }

    # 添加详细的评估指标
    result.update(metrics)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="Dataset to train on")
    parser.add_argument("--model", default="all", help="Model to train")
    args = parser.parse_args()
    
    cfg = read_cfg()
    
    # 确定数据集
    if args.dataset == "all":
        datasets = list(cfg["datasets"].keys())
    else:
        datasets = [args.dataset]
    
    # 确定模型
    if args.model == "all":
        models = ["rf", "xgb", "svr", "lstm", "transformer"]
    else:
        models = [args.model]
    
    # 训练所有组合
    results = []
    for dataset in datasets:
        for model in models:
            try:
                result = train_dataset(dataset, model, cfg)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"训练失败 {dataset} - {model}: {e}")
    
    # 保存训练日志 - 修复追加逻辑
    if results:
        results_df = pd.DataFrame(results)
        log_file = ROOT / "tables" / "train_log_enhanced.csv"

        # 检查文件是否存在
        if log_file.exists():
            # 读取现有数据
            try:
                existing_df = pd.read_csv(log_file)
                # 合并数据，避免重复
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                # 去除可能的重复行（基于dataset和model）
                combined_df = combined_df.drop_duplicates(subset=['dataset', 'model'], keep='last')
                combined_df.to_csv(log_file, index=False)
                print(f"\n✓ 训练完成，共 {len(results)} 个新模型")
                print(f"已追加到现有日志，总计 {len(combined_df)} 个模型")
            except Exception as e:
                print(f"读取现有日志失败: {e}")
                # 如果读取失败，直接覆盖
                results_df.to_csv(log_file, index=False)
                print(f"\n✓ 训练完成，共 {len(results)} 个模型")
        else:
            # 文件不存在，创建新文件
            log_file.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(log_file, index=False)
            print(f"\n✓ 训练完成，共 {len(results)} 个模型")

        print("详细结果保存至 tables/train_log_enhanced.csv")
    else:
        print("没有成功训练的模型")
