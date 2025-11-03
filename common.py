import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

np.random.seed(250)
tf.random.set_seed(250)


# ========== 数据库配置 ==========
def get_db_config():
    """
    从环境变量获取数据库配置
    
    返回:
    - dict: 数据库配置字典
    """
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'AE1_tween_saved_data'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '123456')
    }

# 为了向后兼容，保留 DB_CONFIG 变量
DB_CONFIG = get_db_config()


# ========== 1. 数据读取 ==========
def load_json_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["Items"])


def load_pgsql_data(query=None, table_name=None, conditions=None):
    """
    从 PostgreSQL 数据库加载数据
    
    参数:
    - query: 自定义SQL查询语句（优先级最高）
    - table_name: 表名（如果未提供query，则使用此参数）
    - conditions: WHERE条件（可选，与table_name配合使用）
    
    返回:
    - pandas DataFrame
    """
    try:
        # 建立数据库连接
        conn = psycopg2.connect(**DB_CONFIG)
        
        # 如果提供了自定义查询语句
        if query:
            df = pd.read_sql_query(query, conn)
        # 如果提供了表名
        elif table_name:
            # 指定字段名并使用双引号保留大小写
            if conditions:
                query = f'SELECT "StepName", "Duration", "Timestamp" FROM {table_name} WHERE {conditions}'
            else:
                query = f'SELECT "StepName", "Duration", "Timestamp" FROM {table_name}'
            df = pd.read_sql_query(query, conn)
        else:
            raise ValueError("必须提供 query 或 table_name 参数")
        
        conn.close()
        print(f"✅ 成功从数据库加载 {len(df)} 条记录")
        return df
        
    except psycopg2.Error as e:
        print(f"❌ 数据库连接错误: {e}")
        raise
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise


def load_data(source_type="json", **kwargs):
    """
    统一数据加载接口
    
    参数:
    - source_type: 数据源类型，"json" 或 "pgsql"
    - **kwargs: 
        - 对于 json: file_path
        - 对于 pgsql: query 或 table_name, conditions
    
    返回:
    - pandas DataFrame
    """
    if source_type == "json":
        if 'file_path' not in kwargs:
            raise ValueError("JSON数据源需要提供 file_path 参数")
        return load_json_data(kwargs['file_path'])
    
    elif source_type == "pgsql":
        return load_pgsql_data(
            query=kwargs.get('query'),
            table_name=kwargs.get('table_name'),
            conditions=kwargs.get('conditions')
        )
    
    else:
        raise ValueError(f"不支持的数据源类型: {source_type}，请使用 'json' 或 'pgsql'")


# ========== 2. 数据验证函数 ==========
def validate_data_columns(df, required_columns=None):
    """
    验证数据是否包含必需的列
    
    参数:
    - df: pandas DataFrame
    - required_columns: 必需的列名列表，默认为 ['StepName', 'Duration', 'Timestamp']
    
    返回:
    - True: 验证通过
    
    异常:
    - ValueError: 缺少必需的列
    """
    if required_columns is None:
        required_columns = ['StepName', 'Duration', 'Timestamp']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"❌ 数据缺少必需的列: {missing_columns}\n"
            f"当前列: {list(df.columns)}\n"
            f"必需列: {required_columns}"
        )
    
    print(f"✅ 数据验证通过，包含必需列: {required_columns}")
    return True


# ========== 3. 预处理函数 ==========
def prepare_dataset(df, step_name="zq", seq_len=10):
    """准备训练数据集"""
    # 根据 StepName 过滤数据
    df_filtered = df[df["StepName"] == step_name].copy()
    if df_filtered.empty:
        raise ValueError(f"❌ 步骤 {step_name} 没有数据！")

    # 按 Timestamp 排序（确保时间顺序正确）
    if "Timestamp" in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by="Timestamp").reset_index(drop=True)
    else:
        print("⚠️ 警告：数据中没有 Timestamp 列，无法排序")

    # 清理无效数据
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

    print(f"✅ {step_name}: 准备 {len(X)} 个训练序列")

    return np.array(X), np.array(y), scaler, df_filtered


# ========== 3. 模型构建 ==========
def build_model(input_shape):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(16, activation='tanh', input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
    return model

