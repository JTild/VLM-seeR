import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------- 路径配置 ----------
BASE_DIR = "dataSet/sig_dataset417v2_test"
# BASE_DIR = "./dataSet/sig_dataset"
SUB_DIRS = ["iq_timing", "time_freq", "constellation"]
SUFFIXES = ["_iq.png", "_tf.png", "_const.png"]

# ---------- 模型加载（仅一次） ----------
modelPath = "/home/jql/code/Qwen2.5vl/Qlorav1"
print("正在加载模型，请稍候...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    modelPath, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    modelPath,
    min_pixels=4*28*28,      # 默认值
    max_pixels=784*640       # 与训练时保持一致
    # max_pixels=1200*900,
)
print("模型加载完成，可以开始推理。")

# ---------- 推理函数 ----------
def run_inference(sig_name):
    """sig_name 例如 'sig_000001_msk_snr15.0'"""
    # 自动生成三个路径
    img_paths = [
        os.path.join(BASE_DIR, sub, f"{sig_name}{suf}")
        for sub, suf in zip(SUB_DIRS, SUFFIXES)
    ]

    # 检查文件是否存在
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
    ).to("cuda")

    token_count = inputs.input_ids.shape[1]
    print(f"  -> 输入 token 数量: {token_count}")

    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ---------- 交互循环 ----------
print("\n请输入信号")
print("输入 exit 退出程序\n")

while True:
    user_input = input("信号标识> ").strip()
    if user_input.lower() == 'exit':
        print("退出程序。")
        break
    if not user_input:
        continue

    # 解析多个标识符（逗号或空格分隔）
    if ',' in user_input:
        sig_names = [x.strip() for x in user_input.split(',') if x.strip()]
    else:
        sig_names = user_input.split()

    for sig in sig_names:
        print(f"\n正在处理: {sig}")
        try:
            result = run_inference(sig)
            print(f"模型回复: {result}")
        except Exception as e:
            print(f"处理出错: {e}")