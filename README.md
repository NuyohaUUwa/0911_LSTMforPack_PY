# LSTM 时间序列预测项目

基于 LSTM（长短期记忆网络）的时间序列预测系统，用于预测工业制造过程中各步骤的耗时（Duration）。支持多种数据源、批量训练、预测验证和实时监控功能。

## 📋 项目简介

本项目是一个完整的 LSTM 时间序列预测解决方案，主要用于预测制造流程中各步骤的执行时间。系统支持：

- 🔄 **多种数据源**：JSON 文件和 PostgreSQL 数据库
- 🎯 **批量训练**：支持同时训练多个步骤的模型
- 📊 **预测验证**：自动对比预测值与真实值，生成评估报告
- 🔍 **实时监控**：实时监控数据库，自动进行预测并保存结果
- 📈 **可视化**：自动生成训练损失曲线和预测对比图表

## 🚀 快速开始

### 环境要求

- Python 3.7+
- TensorFlow 2.0+
- PostgreSQL（可选，用于数据库数据源）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量（使用 PostgreSQL 时）

创建 `.env` 文件并配置数据库连接信息：

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=Database
DB_USER=postgres
DB_PASSWORD=password
```

## 📁 项目结构

```
├── common.py              # 公共函数模块（数据加载、预处理、模型构建）
├── train.py              # 模型训练脚本
├── predict.py            # 预测验证脚本
├── predict_and_save.py   # 实时预测与保存脚本
├── main.py               # 主程序（旧版本）
├── requirements.txt      # Python 依赖包
├── .gitignore           # Git 忽略文件配置
├── data/                 # 数据文件目录
│   ├── OnlyPainting_Paint_Data3_Train.json
│   └── OnlyPainting_Paint_Data1_Prediction.json
├── models/               # 训练好的模型保存目录
│   ├── model_拧紧1.keras
│   ├── model_拧紧2.keras
│   └── ...
└── All_plotpics/         # 图表输出目录
    ├── TrainingLoss_*.png
    ├── Prediction_*.png
    └── Validation_*.png
```

## 🔧 使用方法

### 1. 模型训练 (`train.py`)

训练 LSTM 模型用于时间序列预测。

#### 基本配置

在 `train.py` 中修改以下配置：

```python
# 训练轮次（None 表示无限训练，直到手动中断）
train_epochs = 50

# 数据源类型: "json" 或 "pgsql"
data_source = "pgsql"

# 要训练的步骤名称列表
step_names_to_train = [
    "拧紧1",
    "拧紧2",
    "拧紧3",
    "夹紧",
    "松开",
    "取钉1",
    "取钉2"
]

# JSON 数据源配置
train_data_file = "data/OnlyPainting_Paint_Data3_Train.json"

# PostgreSQL 数据源配置
pgsql_table = '"Beats_of_M8_liangainingjin"'
pgsql_query = None  # 或自定义 SQL 查询
```

#### 运行训练

```bash
python train.py
```

#### 训练输出

- 模型文件保存至 `models/model_{步骤名称}.keras`
- 训练损失曲线保存至 `All_plotpics/TrainingLoss_{步骤名称}_{时间戳}.png`

### 2. 预测验证 (`predict.py`)

使用预训练模型进行预测，并与真实值对比验证。

#### 基本配置

在 `predict.py` 中修改以下配置：

```python
# 预测数量
num_of_prediction = 5

# 训练数据数量（None 表示自动计算为 total_count - num_of_prediction）
train_data_count = None

# 数据源类型
data_source = "pgsql"

# 要预测的步骤名称列表
step_names_to_predict = [
    "拧紧1",
    "拧紧2",
    "拧紧3",
    "夹紧",
    "松开",
    "取钉1",
    "取钉2"
]

# 数据文件配置
predict_data_file = "data/OnlyPainting_Paint_Data1_Prediction.json"
pgsql_table = '"Beats_of_M8_liangainingjin"'
```

#### 运行预测验证

```bash
python predict.py
```

#### 验证输出

- 预测对比图表保存至 `All_plotpics/Validation_{步骤名称}_{时间戳}.png`
- 控制台输出预测性能评估指标：
  - MAE（平均绝对误差）
  - MSE（均方误差）
  - RMSE（均方根误差）

### 3. 实时预测与保存 (`predict_and_save.py`)

实时监控数据库，当检测到新数据时自动进行预测并保存结果。

#### 运行实时监控

```bash
python predict_and_save.py
```

#### 功能说明

- 🔍 **自动监控**：每 100ms 查询一次数据库，检测新记录
- 🎯 **自动预测**：检测到新记录时，使用对应步骤的预训练模型进行预测
- 💾 **自动保存**：预测结果自动保存至 `Beats_of_M8_liangainingjin_yuce` 表
- 🆔 **ID 生成**：自动生成 8 位 ID 号（前 5 位 Base36 日期 + 后 3 位序号）
- 🔄 **去重机制**：自动检测并跳过已存在的预测记录

#### ID 号生成规则

- 前 5 位：Base36 编码的日期（YYYYMMDD）
- 后 3 位：递增序号（001, 002, ...）

示例：`KPR50`（日期部分）+ `001`（序号） = `KPR50001`

## 📊 数据格式

### JSON 数据格式

```json
{
  "Items": [
    {
      "StepName": "拧紧1",
      "Duration": 2.5,
      "Timestamp": "2024-01-01T10:00:00"
    },
    ...
  ]
}
```

### 数据库表结构

必需字段：

- `StepName` (TEXT): 步骤名称
- `Duration` (DOUBLE PRECISION): 耗时（秒）
- `Timestamp` (TIMESTAMP): 时间戳

可选字段（用于实时预测）：

- `Address` (TEXT): 地址
- `IDnumber` (CHARACTER VARYING): ID 号

## 🎯 模型架构

### LSTM 模型结构

```
Sequential Model:
├── LSTM Layer
│   ├── Units: 16
│   ├── Activation: tanh
│   ├── Dropout: 0.2
│   └── Recurrent Dropout: 0.2
└── Dense Layer
    └── Units: 1 (输出层)
```

### 训练参数

- **优化器**：Adam (learning_rate=0.0005, clipvalue=1.0)
- **损失函数**：MSE (均方误差)
- **序列长度**：10（默认）
- **数据归一化**：MinMaxScaler (0-1 范围)

### 早停策略

- **完整训练模式**：patience = min(50, max(20, data_size))
- **验证模式**：patience = min(20, max(10, data_size // 2))

## 📈 输出说明

### 训练输出文件

- `models/model_{步骤名称}.keras`: 训练好的模型文件
- `All_plotpics/TrainingLoss_{步骤名称}_{时间戳}.png`: 训练损失曲线

### 预测输出文件

- `All_plotpics/Prediction_{步骤名称}_{时间戳}.png`: 预测结果图表（仅预测）
- `All_plotpics/Validation_{步骤名称}_{时间戳}.png`: 预测验证对比图表（含真实值对比）

### 性能评估指标

- **MAE (Mean Absolute Error)**: 平均绝对误差，单位：秒
- **MSE (Mean Squared Error)**: 均方误差
- **RMSE (Root Mean Squared Error)**: 均方根误差，单位：秒

## ⚙️ 高级配置

### 自定义序列长度

在 `common.py` 的 `prepare_dataset` 函数中修改 `seq_len` 参数（默认值为 10）：

```python
X_train, y_train, scaler, df_filtered = prepare_dataset(
    df_train, 
    step_name=step_name, 
    seq_len=10  # 修改此值
)
```

### 自定义模型架构

在 `common.py` 的 `build_model` 函数中修改模型结构：

```python
def build_model(input_shape):
    model = Sequential([
        LSTM(16, activation='tanh', input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
        # 可以添加更多层
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005, clipvalue=1.0), loss='mse')
    return model
```

### 自定义 SQL 查询

在 `train.py` 或 `predict.py` 中设置 `pgsql_query`：

```python
pgsql_query = """
SELECT "StepName", "Duration", "Timestamp" 
FROM "Beats_of_M8_liangainingjin" 
WHERE "StepName" LIKE '%涂胶%'
ORDER BY "Timestamp"
"""
```

## 🔍 故障排除

### 常见问题

1. **模型文件不存在**
   - 确保已运行 `train.py` 训练模型
   - 检查 `models/` 目录中是否存在对应的模型文件

2. **数据不足**
   - 确保数据量至少大于序列长度（默认 10）
   - 检查数据中是否包含指定的 `StepName`

3. **数据库连接失败**
   - 检查 `.env` 文件中的数据库配置
   - 确保 PostgreSQL 服务正在运行
   - 验证数据库用户权限

4. **预测结果为 NaN 或 Inf**
   - 检查训练数据质量
   - 确认数据中无异常值
   - 尝试重新训练模型

## 📝 注意事项

1. **数据质量**：确保数据按时间戳排序，且无缺失值
2. **模型保存**：训练完成后模型会自动保存，建议定期备份 `models/` 目录
3. **实时监控**：`predict_and_save.py` 会持续运行，使用 Ctrl+C 停止
4. **资源消耗**：实时监控模式会频繁查询数据库，注意数据库性能影响

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
