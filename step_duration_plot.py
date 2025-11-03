
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取 JSON 文件
file_path = "Paint_Data3.json"  # 请替换为你的实际文件名
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 整理数据：按 StepName 分类，收集每个 Duration
step_durations = defaultdict(list)
for item in data["Items"]:
    step_durations[item["StepName"]].append(item["Duration"])

# 准备绘图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))
for step_name, durations in step_durations.items():
    plt.plot(durations, marker='o', label=step_name)

# 设置图表属性
plt.title(f"StepName vs Duration ({file_path})")
plt.xlabel("Occurrence")
plt.ylabel("Duration (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
