import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, date
from tensorflow.keras.models import load_model
from common import load_pgsql_data, prepare_dataset, validate_data_columns, get_db_config
import psycopg2
from psycopg2.extras import RealDictCursor
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ========== 数据库配置 ==========
# 从环境变量获取数据库配置
DB_CONFIG = get_db_config()


# ========== 预测函数 ==========
def predict_latest_values(model, df, scaler, step_name, seq_len=10, predict_count=5):
    """
    使用训练好的模型预测最新值
    
    参数:
    - model: 训练好的模型
    - df: 包含历史数据的 DataFrame
    - scaler: 数据缩放器
    - step_name: 步骤名称
    - seq_len: 序列长度
    - predict_count: 预测数量
    
    返回:
    - predictions: 预测值列表
    """
    # 获取目标步骤的数据
    df_filtered = df[df["StepName"] == step_name].copy()
    
    if df_filtered.empty:
        raise ValueError(f"❌ '{step_name}' 没有找到匹配的记录")
    
    # 按时间排序
    if "Timestamp" in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by="Timestamp").reset_index(drop=True)
    
    # 清理数据
    df_filtered = df_filtered.dropna(subset=["Duration"])
    df_filtered["Duration"] = df_filtered["Duration"].astype(float)
    
    if len(df_filtered) < seq_len:
        raise ValueError(f"❌ '{step_name}' 的有效数据不足 {seq_len} 条")
    
    # 准备数据
    durations = df_filtered["Duration"].values.reshape(-1, 1)
    durations_scaled = scaler.transform(durations)
    
    # 获取最新的序列作为模型输入
    current_seq = durations_scaled[-seq_len:]
    
    # 进行预测
    predictions = []
    for i in range(predict_count):
        input_seq = current_seq.reshape(1, seq_len, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        
        if np.isnan(pred) or np.isinf(pred):
            print(f"⚠️ 第 {i + 1} 次预测结果无效")
            break
            
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)
    
    # 反归一化
    if predictions:
        predictions_array = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return predictions_array.tolist()
    else:
        return []


def base36_encode(num):
    """将数字转换为Base36编码"""
    if num == 0:
        return "0"
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while num > 0:
        result = chars[num % 36] + result
        num //= 36
    return result


def generate_idnumber(timestamp, step_name, last_idnumber=None):
    """
    生成IDnumber
    
    参数:
    - timestamp: 时间戳
    - step_name: 步骤名称
    - last_idnumber: 上一个IDnumber（用于递增）
    
    返回:
    - idnumber: 8位ID号
    """
    # 获取日期（年月日）
    date_str = timestamp.strftime("%Y%m%d")
    date_num = int(date_str)
    
    # 转换为Base36并补0到5位
    date_base36 = base36_encode(date_num)
    date_part = date_base36.zfill(5)
    
    if last_idnumber:
        # 基于上一个IDnumber递增
        last_sequence = int(last_idnumber[-3:])
        sequence_part = str(last_sequence + 1).zfill(3)
    else:
        # 首次生成，从001开始
        sequence_part = "001"
    
    return date_part + sequence_part


def get_latest_record():
    """获取最新记录"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
        SELECT "Address", "StepName", "Duration", "Timestamp", "IDnumber"
        FROM "Beats_of_M8_liangainingjin"
        ORDER BY "Timestamp" DESC
        LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            return {
                'Address': result[0],
                'StepName': result[1],
                'Duration': result[2],
                'Timestamp': result[3],
                'IDnumber': result[4]
            }
        return None
        
    except Exception as e:
        print(f"❌ 获取最新记录失败: {e}")
        return None
    finally:
        if conn:
            conn.close()


def get_all_latest_records():
    """获取所有步骤的最新记录"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
        SELECT DISTINCT ON ("StepName") 
               "Address", "StepName", "Duration", "Timestamp", "IDnumber"
        FROM "Beats_of_M8_liangainingjin"
        ORDER BY "StepName", "Timestamp" DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            return [
                {
                    'Address': row[0],
                    'StepName': row[1],
                    'Duration': row[2],
                    'Timestamp': row[3],
                    'IDnumber': row[4]
                }
                for row in results
            ]
        return []
        
    except Exception as e:
        print(f"❌ 获取最新记录失败: {e}")
        return []
    finally:
        if conn:
            conn.close()


def check_prediction_exists(idnumber, step_name, table_name="Beats_of_M8_liangainingjin_yuce"):
    """检查预测是否已存在（同时检查IDnumber和StepName）"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = f"""
        SELECT COUNT(*) FROM "{table_name}"
        WHERE "IDnumber" = %s AND "StepName" = %s
        """
        cursor.execute(query, (idnumber, step_name))
        count = cursor.fetchone()[0]
        
        return count > 0
        
    except Exception as e:
        print(f"❌ 检查预测存在性失败: {e}")
        return False
    finally:
        if conn:
            conn.close()


def save_prediction_to_db(address, step_name, prediction_value, idnumber, table_name="Beats_of_M8_liangainingjin_yuce"):
    """
    保存单个预测结果到数据库
    
    参数:
    - address: 地址
    - step_name: 步骤名称
    - prediction_value: 预测值
    - idnumber: ID号
    - table_name: 目标表名
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 创建预测结果表（如果不存在）
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            "Address" TEXT,
            "StepName" TEXT,
            "Duration_yuce" DOUBLE PRECISION,
            "Timestamp" TIMESTAMP WITH TIME ZONE,
            "IDnumber" CHARACTER VARYING(8)
        )
        """
        cursor.execute(create_table_sql)
        
        # 插入预测结果
        current_time = datetime.now()
        insert_sql = f"""
        INSERT INTO "{table_name}" ("Address", "StepName", "Duration_yuce", "Timestamp", "IDnumber")
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_sql, (address, step_name, prediction_value, current_time, idnumber))
        
        conn.commit()
        print(f"✅ 成功保存预测结果: {step_name} = {prediction_value:.4f}s (ID: {idnumber})")
        
    except Exception as e:
        print(f"❌ 保存预测结果失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def update_existing_table(step_name, predictions, table_name="Beats_of_M8_liangainingjin"):
    """
    更新现有表，添加预测列
    
    参数:
    - step_name: 步骤名称
    - predictions: 预测值列表
    - table_name: 目标表名
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 检查是否存在预测列，如果不存在则添加
        check_column_sql = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s AND column_name = 'prediction_duration'
        """
        cursor.execute(check_column_sql, (table_name,))
        
        if not cursor.fetchone():
            # 添加预测列
            add_column_sql = f"""
            ALTER TABLE "{table_name}" 
            ADD COLUMN prediction_duration FLOAT,
            ADD COLUMN prediction_timestamp TIMESTAMP
            """
            cursor.execute(add_column_sql)
            print("✅ 已添加预测列到现有表")
        
        # 获取该步骤的最新记录ID
        get_latest_sql = f"""
        SELECT id FROM "{table_name}" 
        WHERE "StepName" = %s 
        ORDER BY "Timestamp" DESC 
        LIMIT 1
        """
        cursor.execute(get_latest_sql, (step_name,))
        result = cursor.fetchone()
        
        if result:
            latest_id = result[0]
            print(f"📊 找到步骤 '{step_name}' 的最新记录 ID: {latest_id}")
            
            # 为每个预测值创建新记录
            for i, pred_value in enumerate(predictions, 1):
                # 计算预测时间（基于最新记录的时间 + 预测间隔）
                future_time = datetime.now() + pd.Timedelta(minutes=i*5)  # 假设每5分钟一个预测
                
                insert_sql = f"""
                INSERT INTO "{table_name}" ("StepName", "Duration", "Timestamp", prediction_duration, prediction_timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_sql, (step_name, pred_value, future_time, pred_value, future_time))
            
            conn.commit()
            print(f"✅ 成功添加 {len(predictions)} 个预测记录到现有表")
        else:
            print(f"❌ 未找到步骤 '{step_name}' 的记录")
            
    except Exception as e:
        print(f"❌ 更新现有表失败: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def run_realtime_prediction():
    """
    运行实时预测循环
    """
    print("🚀 开始实时预测监控...")
    print("📋 配置信息:")
    print(f"   - 查询间隔: 100ms")
    print(f"   - 源数据表: Beats_of_M8_liangainingjin")
    print(f"   - 预测表: Beats_of_M8_liangainingjin_yuce")
    print(f"   - IDnumber规则: 前5位Base36日期 + 后3位序号")
    print("\n按 Ctrl+C 停止监控...")
    
    last_processed_ids = {}  # 记录每个步骤最后处理的ID
    
    try:
        while True:
            # 获取所有步骤的最新记录
            latest_records = get_all_latest_records()
            
            if not latest_records:
                print("⚠️ 未找到任何记录")
                time.sleep(0.1)
                continue
            
            # 处理每个步骤的最新记录
            for record in latest_records:
                step_name = record['StepName']
                current_id = record['IDnumber']
                
                # 检查该步骤是否已处理过
                if step_name in last_processed_ids and current_id == last_processed_ids[step_name]:
                    continue  # 跳过已处理的记录
                
                print(f"\n🔍 检测到新记录: {current_id}")
                print(f"   步骤: {step_name}")
                print(f"   地址: {record['Address']}")
                print(f"   时间: {record['Timestamp']}")
                
                # 生成预测ID
                predict_id = generate_idnumber(
                    record['Timestamp'], 
                    step_name, 
                    current_id
                )
                
                # 检查预测是否已存在
                if check_prediction_exists(predict_id, step_name):
                    print(f"⚠️ 预测 {predict_id} ({step_name}) 已存在，跳过")
                    last_processed_ids[step_name] = current_id
                    continue
                
                print(f"🎯 生成预测ID: {predict_id}")
                
                # 进行预测
                try:
                    model_path = f"models/model_{step_name}.keras"
                    
                    if not os.path.exists(model_path):
                        print(f"⚠️ 模型文件不存在: {model_path}")
                        last_processed_ids[step_name] = current_id
                        continue
                    
                    # 加载模型
                    model = load_model(model_path)
                    
                    # 获取该步骤的历史数据
                    where_condition = f'"StepName" = \'{step_name}\''
                    df_step = load_pgsql_data(
                        query=None,
                        table_name='"Beats_of_M8_liangainingjin"',
                        conditions=where_condition
                    )
                    
                    if df_step is None or len(df_step) < 10:
                        print(f"⚠️ 步骤 '{step_name}' 历史数据不足")
                        last_processed_ids[step_name] = current_id
                        continue
                    
                    # 准备数据
                    X, y, scaler, df_filtered = prepare_dataset(df_step, step_name, 10)
                    
                    if len(X) == 0:
                        print(f"⚠️ 步骤 '{step_name}' 无法准备数据")
                        last_processed_ids[step_name] = current_id
                        continue
                    
                    # 进行预测
                    predictions = predict_latest_values(model, df_filtered, scaler, step_name, 10, 1)
                    
                    if predictions:
                        prediction_value = predictions[0]
                        print(f"📊 预测结果: {prediction_value:.4f}s")
                        
                        # 保存预测结果
                        save_prediction_to_db(
                            address=record['Address'],
                            step_name=step_name,
                            prediction_value=prediction_value,
                            idnumber=predict_id
                        )
                    else:
                        print(f"❌ 预测失败")
                    
                except Exception as e:
                    print(f"❌ 预测过程出错: {e}")
                
                # 更新该步骤最后处理的ID
                last_processed_ids[step_name] = current_id
            
            # 等待100ms
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 监控已停止")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")


# ========== 主程序 ==========
if __name__ == "__main__":
    run_realtime_prediction()
