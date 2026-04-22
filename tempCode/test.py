import json

file_path = "/home/jql/code/LLaMA-Factory/data/qwen_vl_train.jsonl"
errors = []

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON decode error - {e}")
            continue
        # 检查 content 字段类型
        msgs = data.get("messages", [])
        for j, msg in enumerate(msgs):
            content = msg.get("content")
            if not isinstance(content, list):
                errors.append(f"Line {i}, message {j}: content is {type(content).__name__}, expected list")

if errors:
    print("发现错误：")
    for err in errors:
        print(err)
else:
    print("文件格式正确！")