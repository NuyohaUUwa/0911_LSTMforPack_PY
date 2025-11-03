import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from common import load_json_data, prepare_dataset


# ========== 预测函数 ==========
def predict_future(df, model_path, scaler, step_name="zq", seq_len=10, predict_count=10):
    """使用保存的模型文件进行预测"""
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

    for i in range(predict_count):
        input_seq = current_seq.reshape(1, seq_len, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        if np.isnan(pred) or np.isinf(pred):
            print(f"🚫 预测失败: 第 {i + 1} 次预测结果无效")
            break
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    if len(predictions) == 0:
        print("⚠️ 无有效预测结果")
        return [], df_filtered

    predictions_array = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
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

    for i in range(predict_count):
        input_seq = current_seq.reshape(1, seq_len, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        if np.isnan(pred) or np.isinf(pred):
            print(f"🚫 预测失败: 第 {i + 1} 次预测结果无效")
            break
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    if len(predictions) == 0:
        print("⚠️ 无有效预测结果")
        return [], df_filtered

    predictions_array = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions_array, df_filtered


# ========== 可视化函数 ==========
def plot_predictions(df_filtered, pred_durations):
    """绘制预测结果图表"""
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
    
    # 计算评估指标（在设置图表属性之前）
    mae = np.mean(np.abs(pred_durations - true_values))
    mse = np.mean((pred_durations - true_values) ** 2)
    rmse = np.sqrt(mse)
    
    # 设置图表属性
    plt.xlabel("数据点索引")
    plt.ylabel("Duration（秒）")
    plt.title(f"{step_name} 步骤耗时预测验证对比\n训练数据: {len(train_data)}个点, 预测: {len(pred_durations)}个点")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加分界线
    plt.axvline(x=pred_start_idx-0.5, color='gray', linestyle=':', alpha=0.7, label="训练/预测分界线")
    
    # 在图表上添加预测性能评估指标
    metrics_text = f"预测性能评估:\nMAE: {mae:.4f}秒\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}秒"
    plt.text(0.02, 0.98, metrics_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存验证对比图表
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"All_plotpics/Validation_{step_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 验证对比图表已保存: {filename}")
    
    plt.show()
    
    # 同时在控制台打印评估指标
    print(f"\n📊 预测性能评估:")
    print(f"   平均绝对误差 (MAE): {mae:.4f} 秒")
    print(f"   均方误差 (MSE): {mse:.4f}")
    print(f"   均方根误差 (RMSE): {rmse:.4f} 秒")


# ========== 数据库相关函数 ==========
def build_pgsql_conditions(step_names_to_predict):
    """
    根据步骤名称列表生成 PostgreSQL WHERE 条件
    
    参数:
    - step_names_to_predict: 步骤名称列表
    
    返回:
    - SQL WHERE 条件字符串，如果列表为空则返回 None
    """
    if step_names_to_predict:
        step_conditions = "', '".join(step_names_to_predict)
        # 使用双引号包裹字段名以保留大小写
        return f'"StepName" IN (\'{step_conditions}\')'
    else:
        return None


def load_prediction_data(data_source, json_file=None, pgsql_query=None, pgsql_table=None, pgsql_conditions=None):
    """
    统一的数据加载接口，根据数据源类型加载数据
    
    参数:
    - data_source: 数据源类型 ("json" 或 "pgsql")
    - json_file: JSON 文件路径（当 data_source="json" 时使用）
    - pgsql_query: 自定义 SQL 查询语句（当 data_source="pgsql" 时使用）
    - pgsql_table: 表名（当 data_source="pgsql" 时使用）
    - pgsql_conditions: WHERE 条件（当 data_source="pgsql" 时使用）
    
    返回:
    - pandas DataFrame
    """
    print("📂 正在加载预测数据...")
    
    if data_source == "json":
        from common import load_json_data
        if json_file is None:
            raise ValueError("使用 JSON 数据源时必须提供 json_file 参数")
        df = load_json_data(json_file)
        
    elif data_source == "pgsql":
        from common import load_pgsql_data
        df = load_pgsql_data(query=pgsql_query, table_name=pgsql_table, conditions=pgsql_conditions)
        
    else:
        raise ValueError(f"不支持的数据源类型: {data_source}，请使用 'json' 或 'pgsql'")
    
    print("✅ 数据加载完成")
    
    # 验证数据列
    from common import validate_data_columns
    validate_data_columns(df)
    
    # 显示数据概览
    print(f"📊 数据总行数: {len(df)}")
    print(f"📋 包含的 StepName: {df['StepName'].unique().tolist()}")
    print(f"📅 时间范围: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")
    
    return df


def print_prediction_config(num_of_prediction, train_data_count, data_source, step_names_to_predict,
                           json_file=None, pgsql_query=None, pgsql_table=None, pgsql_conditions=None):
    """
    打印预测配置信息
    
    参数:
    - num_of_prediction: 预测数量
    - train_data_count: 训练数据数量
    - data_source: 数据源类型
    - step_names_to_predict: 要预测的步骤列表
    - json_file: JSON 文件路径
    - pgsql_query: SQL 查询语句
    - pgsql_table: 表名
    - pgsql_conditions: WHERE 条件
    """
    print(f"\n🔧 开始全流程预测验证模式...")
    print(f"📋 配置信息:")
    print(f"   - 预测数量: {num_of_prediction}")
    print(f"   - 训练数据数量: {train_data_count}")
    print(f"   - 数据源类型: {data_source}")
    print(f"   - 要预测的步骤: {step_names_to_predict}")
    
    if data_source == "json":
        print(f"   - 数据文件: {json_file}")
        
    elif data_source == "pgsql":
        if pgsql_query:
            print(f"   - SQL查询: {pgsql_query}")
        else:
            print(f"   - 数据表: {pgsql_table}")
            if pgsql_conditions:
                print(f"   - 查询条件: {pgsql_conditions}")
    
    print(f"\n🔧 数据处理流程:")
    print(f"   1. 根据 StepName 分类")
    print(f"   2. 按 Timestamp 排序")
    print(f"   3. 使用 Duration 值进行预测验证")


# ========== 全流程预测验证函数 ==========
def run_full_painting_validation(df_predict, num_of_prediction, train_data_count=None, step_names_to_predict=None):
    """
    全流程预测验证模式：使用预训练模型进行预测验证
    按照用户原始想法：使用预训练模型 + 预测集前a条数据衔接 + 预测后Y条数据对比
    
    参数:
    - df_predict: 预测数据
    - num_of_prediction: 预测数量Y
    - train_data_count: 训练数据数量a，如果为None则自动计算为total_count - num_of_prediction
    - step_names_to_predict: 要预测的步骤名称列表，如果为None则自动识别涂胶步骤
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确定要预测的步骤
    if step_names_to_predict is None:
        # 自动识别所有涂胶步骤（向后兼容）
        all_steps = df_predict['StepName'].unique()
        steps_to_predict = [step for step in all_steps if '涂胶' in step or '下轮涂胶' in step]
        print(f"🎯 自动识别到涂胶步骤: {steps_to_predict}")
    else:
        # 使用用户指定的步骤列表
        steps_to_predict = step_names_to_predict
        print(f"🎯 用户指定要预测的步骤: {steps_to_predict}")
    
    # 检查数据中实际存在的步骤
    available_steps = df_predict['StepName'].unique()
    valid_steps = [step for step in steps_to_predict if step in available_steps]
    missing_steps = [step for step in steps_to_predict if step not in available_steps]
    
    if missing_steps:
        print(f"⚠️ 以下步骤在数据中不存在: {missing_steps}")
    
    if not valid_steps:
        print("❌ 没有找到有效的步骤进行预测！")
        return
    
    print(f"📊 有效预测步骤数: {len(valid_steps)}")
    print(f"📋 可用步骤列表: {valid_steps}")
    
    for step_name in valid_steps:
        print(f"\n{'='*50}")
        print(f"🔧 验证步骤：{step_name}")
        print(f"{'='*50}")
        
        try:
            # 检查预训练模型是否存在
            model_file = f"models/model_{step_name}.keras"
            if not os.path.exists(model_file):
                print(f"❌ 未找到预训练模型")
                continue
            
            # 获取预测数据中该步骤的总数量X
            df_predict_filtered = df_predict[df_predict["StepName"] == step_name].copy()
            if df_predict_filtered.empty:
                print(f"❌ 预测数据中没有找到步骤数据")
                continue
                
            total_count = len(df_predict_filtered)
            
            # 计算训练数据量
            if train_data_count is None:
                # 自动计算：前X-Y个值
                train_count = total_count - num_of_prediction
            else:
                # 用户指定：前a个值
                train_count = train_data_count
                
            if train_count <= 0 or train_count + num_of_prediction > total_count:
                print(f"❌ 数据不足")
                continue
                
            # 使用预测数据的前X-Y个值构建scaler（用于数据预处理）
            df_train_combined = df_predict_filtered.head(train_count)
            _, _, scaler, _ = prepare_dataset(df_train_combined, step_name=step_name)
            
            # 检查并加载预训练模型
            if not os.path.exists(model_file):
                print(f"❌ 未找到预训练模型 {model_file}")
                continue
            
            try:
                # 加载预训练模型并预测
                model = tf.keras.models.load_model(model_file, compile=False)
                pred_values, df2_filtered = predict_future_with_model(model, df_predict, scaler, step_name=step_name, predict_count=num_of_prediction)
                
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                continue

            # 检查预测结果
            if len(pred_values) > 0:
                # 获取指定范围的真实值用于对比
                start_idx = train_count
                end_idx = train_count + num_of_prediction
                true_values = df_predict_filtered["Duration"].iloc[start_idx:end_idx].values
                
                # 绘制对比图表
                plot_predictions_with_validation(df2_filtered, pred_values, true_values, step_name)

        except Exception as e:
            print(f"❌ 处理失败：{e}")


# ========== 主程序 ==========
if __name__ == "__main__":
    # ========== 配置参数 ==========
    num_of_prediction = 5  # 预测数量Y，用于对比验证
    train_data_count = None  # 训练数据数量a，None表示自动计算，指定数字表示使用前a个值
    
    # 👉 数据源配置
    data_source = "pgsql"  # 数据源类型: "json" 或 "pgsql"
    
    # 👉 要预测的 StepName 配置（您可以根据需要修改这个列表）
    step_names_to_predict = [
        "拧紧1",    # Tighten 1
        "拧紧2",    # Tighten 2  
        "拧紧3",    # Tighten 3
        "夹紧",     # Clamp
        "松开",     # Loosen
        "取钉1",    # Remove Nail 1
        "取钉2"     # Remove Nail 2
    ]
    
    # JSON 数据源配置
    predict_data_file = "data/OnlyPainting_Paint_Data1_Prediction.json"  # 预测数据文件
    
    # PostgreSQL 数据源配置（当 data_source = "pgsql" 时使用）
    pgsql_query = None  # 自定义SQL查询语句，例如: "SELECT StepName, Duration, Timestamp FROM Beats_of_M8_liangainingjin WHERE StepName LIKE '%涂胶%'"
    pgsql_table = '"Beats_of_M8_liangainingjin"'  # 表名
    
    # ========== 数据加载和配置 ==========
    # 生成 PostgreSQL WHERE 条件
    pgsql_conditions = build_pgsql_conditions(step_names_to_predict)
    
    # 加载预测数据
    df_predict = load_prediction_data(
        data_source=data_source,
        json_file=predict_data_file,
        pgsql_query=pgsql_query,
        pgsql_table=pgsql_table,
        pgsql_conditions=pgsql_conditions
    )
    
    # 打印配置信息
    print_prediction_config(
        num_of_prediction=num_of_prediction,
        train_data_count=train_data_count,
        data_source=data_source,
        step_names_to_predict=step_names_to_predict,
        json_file=predict_data_file,
        pgsql_query=pgsql_query,
        pgsql_table=pgsql_table,
        pgsql_conditions=pgsql_conditions
    )
    
    run_full_painting_validation(df_predict, num_of_prediction, train_data_count, step_names_to_predict)
    print(f"\n🎉 全流程预测验证完成！")

