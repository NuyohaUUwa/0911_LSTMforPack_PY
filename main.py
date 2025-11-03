import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam  # 正确路径
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from datetime import datetime
from typing import Optional, Dict, Any

np.random.seed(250)
tf.random.set_seed(250)


# ========== 1. 数据读取 ==========
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["Items"])


# ========== 2. 预处理函数 ==========
def prepare_dataset(df, step_name="zq", seq_len=10):
    df_filtered = df[df["StepName"] == step_name].copy()
    if df_filtered.empty:
        raise ValueError(f"❌ 步骤 {step_name} 没有数据！")

    print(f"\n📊 Step = '{step_name}' 原始 Duration 数据:", df_filtered["Duration"].head(10).tolist())
    print("📏 NaN 数量：", df_filtered["Duration"].isna().sum())

    df_filtered = df_filtered.dropna(subset=["Duration"])
    df_filtered["Duration"] = df_filtered["Duration"].astype(float)

    durations = df_filtered["Duration"].values.reshape(-1, 1)
    durations = durations[~np.isnan(durations)]
    durations = durations[~np.isinf(durations)]

    if len(durations) <= seq_len:
        raise ValueError(f"❌ 清洗后数据量不足（{len(durations)} 条），无法生成训练序列")

    scaler = MinMaxScaler()
    durations_scaled = scaler.fit_transform(durations.reshape(-1, 1))

    X, y = [], []
    for i in range(len(durations_scaled) - seq_len):
        X.append(durations_scaled[i:i + seq_len])
        y.append(durations_scaled[i + seq_len])

    print("🧪 示例归一化 X[0]:", X[0].flatten())
    print("🧪 示例归一化 y[0]:", y[0])
    print("📈 Scaler min_:", scaler.data_min_)
    print("📈 Scaler range_:", scaler.data_range_)

    return np.array(X), np.array(y), scaler, df_filtered


# ========== 3. 模型构建 ==========
def build_model(input_shape):
    model = Sequential([
        LSTM(16, activation='tanh', input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
    return model


# ========== 4. 无限循环训练 ==========
def infinite_training(X, y, model_path="model_checkpoint.keras", max_epochs=None, step_name="未知步骤", training_type="full"):
    """
    支持无限训练或固定轮数训练，自动保存模型，并绘制 loss 曲线。
    
    参数：
    - max_epochs=None：无限训练直到手动中断
    - max_epochs=int：固定轮数训练
    """
    model = build_model((X.shape[1], X.shape[2]))

    # 根据训练类型和数据量设置不同的早停策略
    data_size = len(X)
    if training_type == "validation":
        # 验证模式：更激进的早停
        patience = min(20, max(10, data_size // 2))
        min_delta = 0.001
    else:
        # 完整训练模式：更宽松的早停
        patience = min(50, max(20, data_size))
        min_delta = 0.0005
    
    early_stop = EarlyStopping(
        monitor='loss', 
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"🔧 早停设置: patience={patience}, min_delta={min_delta}")

    losses = []
    epoch = 0

    if max_epochs is None:
        # 无限训练（建议Ctrl+C打断）
        try:
            while True:
                history = model.fit(X, y, epochs=1, verbose=0, callbacks=[early_stop])
                epoch += 1
                loss = history.history['loss'][0]
                losses.append(loss)

                if epoch % 10 == 0:
                    print(f"📚 第 {epoch} 次训练，损失: {loss:.6f}")
                    model.save(model_path)
        except KeyboardInterrupt:
            print(f"\n🛑 手动终止训练（第 {epoch} 次）")
    else:
        # 固定次数训练
        print(f"🚀 开始训练（共 {max_epochs} 次）...")
        history = model.fit(X, y, epochs=max_epochs, callbacks=[early_stop], verbose=1)
        losses = history.history['loss']
        print("✅ 固定轮训练完成")

    # 最后保存一次模型
    # 保存模型 - 使用新的Keras格式
    model.save(model_path)
    print(f"📦 模型已保存至 {model_path}")
    
    final_loss = model.evaluate(X, y, verbose=0)
    print(f"📉 最终损失: {final_loss:.6f}")

    # ========== 📈 绘制训练损失曲线 ========== #
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{step_name} 训练过程 Loss 曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存训练损失曲线
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"All_plotpics/TrainingLoss_{step_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 训练损失曲线已保存: {filename}")
    
    plt.show()
    
    return model


# ========== 5. 预测函数 ==========
def predict_future(df, model_path, scaler, step_name="zq", seq_len=10, predict_count=10):
    try:
        model = load_model(model_path)
    except Exception as e:
        raise ValueError(f"❌ 模型加载失败: {str(e)}")
    
    # 获取目标步骤的数据
    df_filtered = df[df["StepName"] == step_name].copy()
    
    # 清理 NaN 值并确保有足够数据
    if df_filtered.empty:
        raise ValueError(f"❌ '{step_name}' 没有找到匹配的记录")
        
    # 移除 Duration 为 NaN 的记录
    df_filtered = df_filtered.dropna(subset=["Duration"])
    
    # 重置索引以便后续使用负索引 [-seq_len:]
    df_filtered.reset_index(drop=True, inplace=True)
    
    # 确保有足够的数据点
    if len(df_filtered) < seq_len:
        raise ValueError(f"❌ '{step_name}' 的有效数据不足 {seq_len} 条，仅有 {len(df_filtered)} 条")
    
    # 转换数据类型并检查异常值
    durations = df_filtered["Duration"].astype(float).values.reshape(-1, 1)
    
    # 添加数据质量检查
    if np.any(np.isnan(durations)):
        # 记录有问题的行索引
        nan_indices = np.where(np.isnan(durations.flatten()))[0]
        problem_timestamps = df_filtered.iloc[nan_indices]["Timestamp"].tolist()
        raise ValueError(
            f"❌ '{step_name}' 数据异常: {len(nan_indices)} 条记录 Duration 值无效\n"
            f"问题记录时间: {problem_timestamps}"
        )
    
    # 缩放数据
    durations_scaled = scaler.transform(durations)
    
    # 获取最新的序列作为模型输入
    current_seq = durations_scaled[-seq_len:]
    
    # 创建预测存储列表
    predictions = []
    
    # 添加诊断日志（可选）
    print(f"✅ 成功获取 {len(df_filtered)} 条 '{step_name}' 有效记录")
    print(f"最新{seq_len}条序列时间范围: {df_filtered['Timestamp'].iloc[-seq_len]} - {df_filtered['Timestamp'].iloc[-1]}")

    for i in range(predict_count):
        input_seq = current_seq.reshape(1, seq_len, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        print(f"🔍 原始预测值（归一化）: {pred}")
        if np.isnan(pred) or np.isinf(pred):
            print(f"🚫 预测失败: ❌ 第 {i + 1} 次预测结果为 NaN/inf，请检查模型或训练数据")
            break
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    if len(predictions) == 0:
        print("⚠️ 无有效预测结果，终止后续处理。")
        return [], df_filtered

    predictions_array = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    print("🔮 预测值列表（反归一化后）:")
    for i, val in enumerate(predictions_array, 1):
        print(f"预测{i}: {val:.4f} 秒")

    return predictions_array, df_filtered


def predict_future_with_model(model, df, scaler, step_name="zq", seq_len=10, predict_count=10):
    """使用模型对象进行预测，避免文件加载问题"""
    # 获取目标步骤的数据
    df_filtered = df[df["StepName"] == step_name].copy()
    
    # 清理 NaN 值并确保有足够数据
    if df_filtered.empty:
        raise ValueError(f"❌ '{step_name}' 没有找到匹配的记录")
        
    # 移除 Duration 为 NaN 的记录
    df_filtered = df_filtered.dropna(subset=["Duration"])
    
    # 重置索引以便后续使用负索引 [-seq_len:]
    df_filtered.reset_index(drop=True, inplace=True)
    
    # 确保有足够的数据点
    if len(df_filtered) < seq_len:
        raise ValueError(f"❌ '{step_name}' 的有效数据不足 {seq_len} 条，仅有 {len(df_filtered)} 条")
    
    # 转换数据类型并检查异常值
    durations = df_filtered["Duration"].astype(float).values.reshape(-1, 1)
    
    # 添加数据质量检查
    if np.any(np.isnan(durations)):
        # 记录有问题的行索引
        nan_indices = np.where(np.isnan(durations.flatten()))[0]
        problem_timestamps = df_filtered.iloc[nan_indices]["Timestamp"].tolist()
        raise ValueError(
            f"❌ '{step_name}' 数据异常: {len(nan_indices)} 条记录 Duration 值无效\n"
            f"问题记录时间: {problem_timestamps}"
        )
    
    # 缩放数据
    durations_scaled = scaler.transform(durations)
    
    # 获取最新的序列作为模型输入
    current_seq = durations_scaled[-seq_len:]
    
    # 创建预测存储列表
    predictions = []
    
    # 添加诊断日志（可选）
    print(f"✅ 成功获取 {len(df_filtered)} 条 '{step_name}' 有效记录")
    print(f"最新{seq_len}条序列时间范围: {df_filtered['Timestamp'].iloc[-seq_len]} - {df_filtered['Timestamp'].iloc[-1]}")

    for i in range(predict_count):
        input_seq = current_seq.reshape(1, seq_len, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        print(f"🔍 原始预测值（归一化）: {pred}")
        if np.isnan(pred) or np.isinf(pred):
            print(f"🚫 预测失败: ❌ 第 {i + 1} 次预测结果为 NaN/inf，请检查模型或训练数据")
            break
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    if len(predictions) == 0:
        print("⚠️ 无有效预测结果，终止后续处理。")
        return [], df_filtered

    predictions_array = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    print("🔮 预测值列表（反归一化后）:")
    for i, val in enumerate(predictions_array, 1):
        print(f"预测{i}: {val:.4f} 秒")

    return predictions_array, df_filtered


# ========== 6. 可视化 ==========
def plot_predictions(df_filtered, pred_durations):
    plt.figure(figsize=(16, 6))
    plt.plot(df_filtered["Duration"].values, label="历史数据")
    plt.plot(range(len(df_filtered), len(df_filtered) + len(pred_durations)), pred_durations, "o-", color="orange",
             label="预测")

    for i, v in enumerate(df_filtered["Duration"].values):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(pred_durations):
        plt.text(len(df_filtered) + i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color="orange")

    labels = df_filtered["StepName"].tolist() + [f"预测{i + 1}" for i in range(len(pred_durations))]
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.xlabel("步骤")
    plt.ylabel("Duration（秒）")
    plt.title(f"{df_filtered.iloc[0]['StepName']} 步骤耗时预测")
    plt.legend()
    plt.tight_layout()
    
    # 保存预测结果图表
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    step_name = df_filtered.iloc[0]['StepName']
    filename = f"All_plotpics/Prediction_{step_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 预测结果图表已保存: {filename}")
    
    plt.show()


def plot_predictions_with_validation(df_filtered, pred_durations, true_values, step_name):
    """绘制包含验证对比的预测图表"""
    plt.figure(figsize=(18, 8))
    
    # 训练数据（历史数据）
    train_data = df_filtered["Duration"].values
    plt.plot(train_data, "b-", label="训练数据", linewidth=2, marker='o', markersize=4)
    
    # 预测值
    pred_start_idx = len(train_data)
    pred_indices = range(pred_start_idx, pred_start_idx + len(pred_durations))
    plt.plot(pred_indices, pred_durations, "r-", label="预测值", linewidth=2, marker='s', markersize=6)
    
    # 真实值（用于验证）
    true_indices = range(pred_start_idx, pred_start_idx + len(true_values))
    plt.plot(true_indices, true_values, "g-", label="真实值", linewidth=2, marker='^', markersize=6)
    
    # 添加数值标签
    for i, v in enumerate(train_data):
        plt.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8, color="blue")
    
    for i, v in enumerate(pred_durations):
        plt.text(pred_start_idx + i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8, color="red")
    
    for i, v in enumerate(true_values):
        plt.text(pred_start_idx + i, v - 0.1, f"{v:.2f}", ha="center", fontsize=8, color="green")
    
    # 添加误差线
    for i, (pred, true) in enumerate(zip(pred_durations, true_values)):
        plt.plot([pred_start_idx + i, pred_start_idx + i], [pred, true], 
                "k--", alpha=0.5, linewidth=1)
        error = abs(pred - true)
        plt.text(pred_start_idx + i, (pred + true) / 2, f"误差:{error:.3f}", 
                ha="center", fontsize=7, color="purple")
    
    # 设置图表属性
    plt.xlabel("数据点索引")
    plt.ylabel("Duration（秒）")
    plt.title(f"{step_name} 步骤耗时预测验证对比\n训练数据: {len(train_data)}个点, 预测: {len(pred_durations)}个点")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加分界线
    plt.axvline(x=pred_start_idx-0.5, color='gray', linestyle=':', alpha=0.7, label="训练/预测分界线")
    
    plt.tight_layout()
    
    # 保存验证对比图表
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"All_plotpics/Validation_{step_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 验证对比图表已保存: {filename}")
    
    plt.show()
    
    # 计算并显示评估指标
    mae = np.mean(np.abs(pred_durations - true_values))
    mse = np.mean((pred_durations - true_values) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 预测性能评估:")
    print(f"   平均绝对误差 (MAE): {mae:.4f} 秒")
    print(f"   均方误差 (MSE): {mse:.4f}")
    print(f"   均方根误差 (RMSE): {rmse:.4f} 秒")


# ========== 7. 字体配置 ==========


# ========== 8. 全流程训练函数 ==========
def run_full_painting_workflow(df_train, train_epochs):
    """
    全流程训练模式：自动识别并训练所有涂胶步骤
    
    参数:
    - df_train: 训练数据
    - train_epochs: 训练轮次
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 自动识别所有涂胶步骤
    all_steps = df_train['StepName'].unique()
    painting_steps = [step for step in all_steps if '涂胶' in step or '下轮涂胶' in step]
    
    print(f"🎯 自动识别到涂胶步骤: {painting_steps}")
    print(f"📊 总步骤数: {len(painting_steps)}")
    
    for step_name in painting_steps:
        print(f"\n\n================= 🚀 全流程训练：{step_name} =================")
        
        try:
            # 使用全部训练数据进行训练
            X_train, y_train, scaler, df_filtered = prepare_dataset(df_train, step_name=step_name)
            
            print(f"📊 训练数据中 {step_name} 步骤总数量: {len(df_filtered)}")
            print(f"📚 使用全部 {len(df_filtered)} 个数据点进行完整训练...")
            
            model_file = f"model_{step_name}.keras"
            
            # 完整训练模式：使用全部数据训练
            print(f"🚀 开始完整训练（共 {train_epochs} 轮）...")
            model = infinite_training(X_train, y_train, model_path=model_file, max_epochs=train_epochs, step_name=step_name, training_type="full")
            
            print(f"✅ {step_name} 预训练模型已保存至 {model_file}")
            print(f"📈 训练数据量: {len(df_filtered)} 条")
            print(f"🎯 模型可用于后续预测任务")

        except Exception as e:
            print(f"❌ 步骤 {step_name} 训练失败：{e}")


def run_full_painting_validation(df_predict, num_of_prediction, train_data_count=None):
    """
    全流程预测验证模式：使用预训练模型进行预测验证
    按照用户原始想法：使用预训练模型 + 预测集前a条数据衔接 + 预测后Y条数据对比
    
    参数:
    - df_predict: 预测数据
    - num_of_prediction: 预测数量Y
    - train_data_count: 训练数据数量a，如果为None则自动计算为total_count - num_of_prediction
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 自动识别所有涂胶步骤
    all_steps = df_predict['StepName'].unique()
    painting_steps = [step for step in all_steps if '涂胶' in step or '下轮涂胶' in step]
    
    print(f"🎯 自动识别到涂胶步骤: {painting_steps}")
    print(f"📊 总步骤数: {len(painting_steps)}")
    
    for step_name in painting_steps:
        print(f"\n\n================= 🔧 全流程验证：{step_name} =================")
        
        try:
            # 检查预训练模型是否存在
            model_file = f"model_{step_name}.keras"
            if not os.path.exists(model_file):
                print(f"❌ 未找到预训练模型 {model_file}，请先运行完整训练模式！")
                continue
            
            # 获取预测数据中该步骤的总数量X
            df_predict_filtered = df_predict[df_predict["StepName"] == step_name].copy()
            if df_predict_filtered.empty:
                print(f"❌ 预测数据中没有找到步骤 {step_name} 的数据！")
                continue
                
            total_count = len(df_predict_filtered)
            print(f"📊 预测数据中 {step_name} 步骤总数量: {total_count}")
            
            # 计算训练数据量
            if train_data_count is None:
                # 自动计算：前X-Y个值
                train_count = total_count - num_of_prediction
            else:
                # 用户指定：前a个值
                train_count = train_data_count
                
            if train_count <= 0:
                print(f"❌ 训练数据不足！指定数量 {train_count} 无效")
                continue
                
            if train_count + num_of_prediction > total_count:
                print(f"❌ 数据不足！需要 {train_count + num_of_prediction} 条，但只有 {total_count} 条")
                continue
                
            print(f"📚 使用前 {train_count} 个值作为输入序列，预测第 {train_count+1} 到第 {train_count+num_of_prediction} 个值")
            
            # 使用预测数据的前X-Y个值构建scaler（用于数据预处理）
            df_train_combined = df_predict_filtered.head(train_count)
            _, _, scaler, _ = prepare_dataset(df_train_combined, step_name=step_name)
            
            # 检查并加载预训练模型
            if not os.path.exists(model_file):
                print(f"❌ 未找到预训练模型 {model_file}")
                print(f"💡 请先运行完整训练模式生成预训练模型")
                continue
            
            try:
                # 加载预训练模型
                print(f"🔄 加载预训练模型: {model_file}")
                model = tf.keras.models.load_model(model_file, compile=False)
                print(f"✅ 预训练模型加载成功")
                
                # 使用预测集前a条数据衔接预训练模型
                print(f"🔗 使用前 {train_count} 条数据衔接预训练模型...")
                
                # 直接使用预训练模型进行预测，不需要重新训练
                pred_values, df2_filtered = predict_future_with_model(model, df_predict, scaler, step_name=step_name, predict_count=num_of_prediction)
                
            except Exception as e:
                print(f"❌ 预训练模型加载失败: {e}")
                print(f"💡 请检查模型文件 {model_file} 是否完整")
                continue

            # 检查预测结果
            if len(pred_values) > 0:
                # 获取指定范围的真实值用于对比
                start_idx = train_count
                end_idx = train_count + num_of_prediction
                true_values = df_predict_filtered["Duration"].iloc[start_idx:end_idx].values
                
                print(f"\n🔍 预测值 vs 真实值对比:")
                print(f"📊 预测范围: 第{start_idx+1}到第{end_idx}个数据点")
                for i in range(len(pred_values)):
                    actual_idx = start_idx + i + 1
                    print(f"第{actual_idx}个: 预测={pred_values[i]:.4f}s, 真实={true_values[i]:.4f}s, 误差={abs(pred_values[i]-true_values[i]):.4f}s")
                
                # 绘制对比图表
                plot_predictions_with_validation(df2_filtered, pred_values, true_values, step_name)

        except Exception as e:
            print(f"❌ 步骤 {step_name} 处理失败：{e}")


# ========== 9. 主控制器 ==========
if __name__ == "__main__":
    # ========== 配置参数 ==========
    # 👉 运行模式设置
    run_mode = "validation"  # "full_training"=完整训练模式, "validation"=预测验证模式
    train_epochs = 50  # 训练轮次，可根据需要调整
    
    # 👉 预测验证设置（仅在validation模式下使用）
    num_of_prediction = 5  # 预测数量Y，用于对比验证
    train_data_count = 15  # 训练数据数量a，None表示自动计算，指定数字表示使用前a个值
    
    # ========== 数据加载 ==========
    print("📂 正在加载数据...")
    df_train = load_json_data("OnlyPainting_Paint_Data3_Train.json")
    df_predict = load_json_data("OnlyPainting_Paint_Data1_Prediction.json")
    print("✅ 数据加载完成")
    
    # ========== 根据模式运行 ==========
    if run_mode == "full_training":
        print(f"\n🚀 开始全流程训练模式...")
        print(f"📋 配置信息:")
        print(f"   - 运行模式: 全流程训练")
        print(f"   - 训练轮次: {train_epochs}")
        print(f"   - 数据源: OnlyPainting_Paint_Data3_Train.json")
        
        run_full_painting_workflow(df_train, train_epochs)
        print(f"\n🎉 全流程训练完成！所有预训练模型已保存。")
        
    elif run_mode == "validation":
        print(f"\n🔧 开始全流程预测验证模式...")
        print(f"📋 配置信息:")
        print(f"   - 运行模式: 全流程预测验证")
        print(f"   - 预测数量: {num_of_prediction}")
        print(f"   - 数据源: OnlyPainting_Paint_Data1_Prediction.json")
        
        run_full_painting_validation(df_predict, num_of_prediction, train_data_count)
        print(f"\n🎉 全流程预测验证完成！")
        
    else:
        print(f"❌ 无效的运行模式: {run_mode}")
        print(f"   请选择: 'full_training' 或 'validation'")
