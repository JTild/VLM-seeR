import pandas as pd
import json
import os

# ==================== 配置部分 ====================
DATASET_ROOT = "./sig_dataset417v2"          # 数据集根目录
OUTPUT_JSONL = "./sig_dataset417v2/qwen_vl_train.jsonl"  # 输出 JSONL 文件路径
# 图片路径前缀（若希望使用绝对路径则设为 DATASET_ROOT 的绝对路径；相对路径也可）
IMAGE_BASE_PATH = os.path.abspath(DATASET_ROOT)

# 用户提示文本（可根据需要修改）
USER_PROMPT = "这些图片分别是一个信号的iq波形图，时频图和星座图，根据图片判断这个信号的干扰类型。"

# ==================== 主程序 ====================
def main():
    meta_path = os.path.join(DATASET_ROOT, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到元数据文件: {meta_path}")

    meta = pd.read_csv(meta_path)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, row in meta.iterrows():
            mod_type = row["mod_type"]

            # 构造三张图片的完整路径
            iq_img = os.path.join(IMAGE_BASE_PATH, "iq_timing", row["iq_image"])
            tf_img = os.path.join(IMAGE_BASE_PATH, "time_freq", row["tf_image"])
            const_img = os.path.join(IMAGE_BASE_PATH, "constellation", row["const_image"])

            # 检查图片是否存在（可选）
            for img_path in (iq_img, tf_img, const_img):
                if not os.path.exists(img_path):
                    print(f"警告: 图片不存在 - {img_path}")

            # 构造单条对话数据
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": iq_img},
                            {"type": "image", "image": tf_img},
                            {"type": "image", "image": const_img},
                            {"type": "text", "text": USER_PROMPT}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": mod_type
                    }
                ]
            }

            # 写入一行 JSON
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"成功生成 {len(meta)} 条数据，保存至 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()