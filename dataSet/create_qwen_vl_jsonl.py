import pandas as pd
import json
import os

# ==================== 配置部分 ====================
DATASET_ROOT = "./sig_dataset417v3_test"                     # 数据集根目录
# OUTPUT_JSONL = "./sig_dataset417v2/qwen_vl_test.jsonl" # 输出文件
OUTPUT_JSONL = DATASET_ROOT+"/qwen_vl_testv3.jsonl" # 输出文件
IMAGE_BASE_PATH = os.path.abspath(DATASET_ROOT)         # 图片绝对路径前缀

# 用户提示文本（如需在文本中指定图片位置，可使用 <image> 占位符）
# Qwen2.5-VL 会自动按顺序将 images 中的图片与 <image> 对应，不写占位符则默认放在最前面
USER_PROMPT = "<image><image><image>这些图片分别是一个信号的iq波形图，时频图和星座图，根据图片判断这个信号的调制类型。"

# ==================== 主程序 ====================
def main():
    meta_path = os.path.join(DATASET_ROOT, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到元数据文件: {meta_path}")

    meta = pd.read_csv(meta_path)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, row in meta.iterrows():
            mod_type = row["mod_type"]

            # 三张图片的绝对路径
            iq_img = os.path.join(IMAGE_BASE_PATH, "iq_timing", row["iq_image"])
            tf_img = os.path.join(IMAGE_BASE_PATH, "time_freq", row["tf_image"])
            const_img = os.path.join(IMAGE_BASE_PATH, "constellation", row["const_image"])

            # 可选：检查图片是否存在
            for img in (iq_img, tf_img, const_img):
                if not os.path.exists(img):
                    print(f"警告: 图片不存在 - {img}")

            # 构造新格式的记录
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": USER_PROMPT               # 纯文本，可包含 <image> 占位符
                    },
                    {
                        "role": "assistant",
                        "content": mod_type                  # 助手回答也是纯文本
                    }
                ],
                "images": [iq_img, tf_img, const_img]        # 图片路径列表
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"成功生成 {len(meta)} 条数据，保存至 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()