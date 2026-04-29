import os
import time   # <--- 新增
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------- 路径配置 ----------
BASE_DIR = "dataSet/sig_dataset417v2_test"
SUB_DIRS = ["iq_timing", "time_freq", "constellation"]
SUFFIXES = ["_iq.png", "_tf.png", "_const.png"]

# ---------- 两模型路径 ----------
MODEL_PATH_A = "/home/jql/code/Qwen2.5vl/Qlorav1"   # 微调模型
MODEL_PATH_B = "/home/jql/code/Qwen2.5vl/Ori"       # 原始模型

# ---------- 模型加载 ----------
print("正在加载微调模型 (A)...")
model_A = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH_A, torch_dtype="auto", device_map="auto"
)
processor_A = AutoProcessor.from_pretrained(
    MODEL_PATH_A,
    min_pixels=4*28*28,
    max_pixels=784*640
)
print("微调模型加载完成。")

print("正在加载原始模型 (B)...")
model_B = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH_B, torch_dtype="auto", device_map="auto"
)
processor_B = AutoProcessor.from_pretrained(
    MODEL_PATH_B,
    min_pixels=4*28*28,
    max_pixels=784*640
)
print("原始模型加载完成。")

# ---------- 推理函数（增加计时）----------
def run_inference(model, processor, sig_name, model_label=""):
    """sig_name 例如 'sig_000001_msk_snr15.0'"""
    img_paths = [
        os.path.join(BASE_DIR, sub, f"{sig_name}{suf}")
        for sub, suf in zip(SUB_DIRS, SUFFIXES)
    ]

    for p in img_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"信号不存在: {p}")

    content = []
    for path in img_paths:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": "这些图片分别是一个信号的iq波形图，时频图和星座图，根据图片判断这个信号的调制类型"})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    token_count = inputs.input_ids.shape[1]
    print(f"  [{model_label}] 输入 token 数量: {token_count}")

    # ---------- 开始推理计时 ----------
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"  [{model_label}] 推理耗时: {elapsed:.2f} 秒")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ---------- 交互循环 ----------
print("\n请输入信号标识。")
print("每个信号将同时由 微调模型(A) 和 原始模型(B) 推理并对比。\n")

while True:
    user_input = input("信号标识> ").strip()
    if user_input.lower() == 'exit':
        print("退出程序。")
        break
    if not user_input:
        continue

    if ',' in user_input:
        sig_names = [x.strip() for x in user_input.split(',') if x.strip()]
    else:
        sig_names = user_input.split()

    for sig in sig_names:
        print(f"\n======== 正在处理: {sig} ========")
        try:
            # 模型 A 推理
            result_A = run_inference(model_A, processor_A, sig, model_label="A(微调)")
            print(f"[模型 A 回复]: {result_A}")

            # 模型 B 推理
            result_B = run_inference(model_B, processor_B, sig, model_label="B(原始)")
            print(f"[模型 B 回复]: {result_B}")

        except Exception as e:
            print(f"处理出错: {e}")