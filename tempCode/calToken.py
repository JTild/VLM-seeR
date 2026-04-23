from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

modelPath = "/home/shared/Qwen2.5vl/Qwen2___5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(modelPath)
image_paths = [
	"../dataSet/sig_dataset417v2/constellation/sig_000001_msk_snr11.4_const.png",
	"../dataSet/sig_dataset417v2/iq_timing/sig_000001_msk_snr11.4_iq.png",
	"../dataSet/sig_dataset417v2/time_freq/sig_000001_msk_snr11.4_tf.png",
]
content = []
for img in image_paths:
    content.append({"type": "image", "image": img})
content.append({"type": "text", "text": "这些图片分别是一个信号的iq波形图，时频图和星座图，根据图片判断这个信号的调制类型"})

messages = [{"role": "user", "content": content}]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("实际送入模型的文本序列：")
print(text)  # 这是 apply_chat_template 返回的字符串

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=False,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# ------------------- 计算 token 数量 -------------------
# 总输入 token 数（包含图片占位符，此时视觉 token 尚未展开）
total_tokens = inputs.input_ids.shape[1]

# 文字 token 数量 = 总 token 数 - 图片占位符数量（每张图片在 input_ids 中占 1 个位置）

input_ids = inputs["input_ids"][0]
image_tokens = (input_ids == 151655).sum().item()  # 151655 是 Qwen2.5-VL 的图片token ID

print(f"图片占用 Token 数: {image_tokens}")
print(f"文本占用 Token 数: {len(input_ids) - image_tokens}")
# 视觉 token 数量：需要从 image_grid_thw 计算
# image_grid_thw 是一个形状为 [num_images, 3] 的张量，三列分别代表 (temporal, height, width)

# ------------------------------------------------------

	
