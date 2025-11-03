import json
import re
import datetime
from pathlib import Path

# 预编译规则
PATTERNS = [
    # "放置1/2/3/5/D/B/9" -> "放置"
    (re.compile(r'^放置[1235DB9]$'), '放置'),

    # "1回原位/2回原位/3回原位" -> "回原位"
    (re.compile(r'^[123]回原位$'), '回原位'),

    # "下轮涂胶1" 到 "下轮涂胶13" -> "下轮涂胶"
    (re.compile(r'^下轮涂胶(1[0-3]|[1-9])$'), '下轮涂胶'),

    # "涂胶【任意数字】-1/2/3" -> "涂胶1/2/3"
    (re.compile(r'^涂胶\d+-(\d)$'), r'涂胶\1'),
]

# 直接在这里修改输入/输出文件
in_file = "Paint_Data2.json"
today_str = datetime.datetime.now().strftime("%m%d")
out_file = f"{Path(in_file).stem}_regulared_{today_str}.json"

with open(in_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data.get('Items', []):
    step = str(item.get('StepName', '') or '')
    for pattern, repl in PATTERNS:
        step = pattern.sub(repl, step)
    item['StepName'] = step

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"处理完成：{in_file} -> {out_file}")
