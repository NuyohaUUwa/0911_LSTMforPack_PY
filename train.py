import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from common import load_json_data, prepare_dataset, build_model


# ========== 训练函数 ==========
def infinite_training(X, y, model_path="models/model_checkpoint.keras", max_epochs=None, step_name="未知步骤", training_type="full"):
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


# ========== 全流程训练函数 ==========
def run_full_painting_workflow(df_train, train_epochs, step_names_to_train=None):
    """
    全流程训练模式：训练指定的步骤
    
    参数:
    - df_train: 训练数据
    - train_epochs: 训练轮次
    - step_names_to_train: 要训练的步骤名称列表，如果为None则自动识别涂胶步骤
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 确定要训练的步骤
    if step_names_to_train is None:
        # 自动识别所有涂胶步骤（向后兼容）
        all_steps = df_train['StepName'].unique()
        steps_to_train = [step for step in all_steps if '涂胶' in step or '下轮涂胶' in step]
        print(f"🎯 自动识别到涂胶步骤: {steps_to_train}")
    else:
        # 使用用户指定的步骤列表
        steps_to_train = step_names_to_train
        print(f"🎯 用户指定要训练的步骤: {steps_to_train}")
    
    # 检查数据中实际存在的步骤
    available_steps = df_train['StepName'].unique()
    valid_steps = [step for step in steps_to_train if step in available_steps]
    missing_steps = [step for step in steps_to_train if step not in available_steps]
    
    if missing_steps:
        print(f"⚠️ 以下步骤在数据中不存在: {missing_steps}")
    
    if not valid_steps:
        print("❌ 没有找到有效的步骤进行训练！")
        return
    
    print(f"📊 有效训练步骤数: {len(valid_steps)}")
    print(f"📋 可用步骤列表: {valid_steps}")
    
    for step_name in valid_steps:
        print(f"\n\n================= 🚀 全流程训练：{step_name} =================")
        
        try:
            # 使用全部训练数据进行训练
            X_train, y_train, scaler, df_filtered = prepare_dataset(df_train, step_name=step_name)
            
            print(f"📊 训练数据中 {step_name} 步骤总数量: {len(df_filtered)}")
            print(f"📚 使用全部 {len(df_filtered)} 个数据点进行完整训练...")
            
            model_file = f"models/model_{step_name}.keras"
            
            # 完整训练模式：使用全部数据训练
            print(f"🚀 开始完整训练（共 {train_epochs} 轮）...")
            model = infinite_training(X_train, y_train, model_path=model_file, max_epochs=train_epochs, step_name=step_name, training_type="full")
            
            print(f"✅ {step_name} 预训练模型已保存至 {model_file}")
            print(f"📈 训练数据量: {len(df_filtered)} 条")
            print(f"🎯 模型可用于后续预测任务")

        except Exception as e:
            print(f"❌ 步骤 {step_name} 训练失败：{e}")


# ========== 数据库相关函数 ==========
def build_pgsql_conditions(step_names_to_train):
    """
    根据步骤名称列表生成 PostgreSQL WHERE 条件
    
    参数:
    - step_names_to_train: 步骤名称列表
    
    返回:
    - SQL WHERE 条件字符串，如果列表为空则返回 None
    """
    if step_names_to_train:
        step_conditions = "', '".join(step_names_to_train)
        # 使用双引号包裹字段名以保留大小写
        return f'"StepName" IN (\'{step_conditions}\')'
    else:
        return None


def load_training_data(data_source, json_file=None, pgsql_query=None, pgsql_table=None, pgsql_conditions=None):
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
    print("📂 正在加载训练数据...")
    
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


def print_training_config(train_epochs, data_source, step_names_to_train, 
                         json_file=None, pgsql_query=None, pgsql_table=None, pgsql_conditions=None):
    """
    打印训练配置信息
    
    参数:
    - train_epochs: 训练轮次
    - data_source: 数据源类型
    - step_names_to_train: 要训练的步骤列表
    - json_file: JSON 文件路径
    - pgsql_query: SQL 查询语句
    - pgsql_table: 表名
    - pgsql_conditions: WHERE 条件
    """
    print(f"\n🚀 开始全流程训练模式...")
    print(f"📋 配置信息:")
    print(f"   - 训练轮次: {train_epochs}")
    print(f"   - 数据源类型: {data_source}")
    print(f"   - 要训练的步骤: {step_names_to_train}")
    
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
    print(f"   3. 使用 Duration 值训练 LSTM 模型")


# ========== 主程序 ==========
if __name__ == "__main__":
    # ========== 配置参数 ==========
    train_epochs = None  # 训练轮次，可根据需要调整
    
    # 👉 数据源配置
    data_source = "pgsql"  # 数据源类型: "json" 或 "pgsql"
    
    # 👉 要训练的 StepName 配置（您可以根据需要修改这个列表）
    step_names_to_train = [
        "拧紧1",    # Tighten 1
        "拧紧2",    # Tighten 2  
        "拧紧3",    # Tighten 3
        "夹紧",     # Clamp
        "松开",     # Loosen
        "取钉1",    # Remove Nail 1
        "取钉2"     # Remove Nail 2
    ]
    
    # JSON 数据源配置
    train_data_file = "data/OnlyPainting_Paint_Data3_Train.json"  # 训练数据文件
    
    # PostgreSQL 数据源配置（当 data_source = "pgsql" 时使用）
    pgsql_query = None  # 自定义SQL查询语句，例如: "SELECT StepName, Duration, Timestamp FROM Beats_of_M8_liangainingjin WHERE StepName LIKE '%涂胶%'"
    pgsql_table = '"Beats_of_M8_liangainingjin"'  # 表名
    
    # ========== 数据加载和配置 ==========
    # 生成 PostgreSQL WHERE 条件
    pgsql_conditions = build_pgsql_conditions(step_names_to_train)
    
    # 加载训练数据
    df_train = load_training_data(
        data_source=data_source,
        json_file=train_data_file,
        pgsql_query=pgsql_query,
        pgsql_table=pgsql_table,
        pgsql_conditions=pgsql_conditions
    )
    
    # 打印配置信息
    print_training_config(
        train_epochs=train_epochs,
        data_source=data_source,
        step_names_to_train=step_names_to_train,
        json_file=train_data_file,
        pgsql_query=pgsql_query,
        pgsql_table=pgsql_table,
        pgsql_conditions=pgsql_conditions
    )
    
    run_full_painting_workflow(df_train, train_epochs, step_names_to_train)
    print(f"\n🎉 全流程训练完成！所有预训练模型已保存。")

