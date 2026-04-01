import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# 你的模型路径
# model_dir = "/home/shared/models/Qwen/Qwen3___5-9B"
model_dir = "/home/jql/.cache/modelscope/hub/models/DavidWen2025"


# 1. 加载处理器（关键：指定 trust_remote_code 加载多模态处理器）
processor = AutoProcessor.from_pretrained(
    model_dir,
    trust_remote_code=True,
    padding_side="left"  # 多模态推理推荐配置
)

# 2. 加载模型（关键：显式启用 remote code 加载自定义架构）
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,  # 必须启用，否则无法识别多模态架构
    low_cpu_mem_usage=True   # 降低CPU内存占用
)

# 3. 准备输入
image_path = "R-C.jpeg"
image = Image.open(image_path).convert('RGB')
prompt = "你支持图片输入吗？"

# 多模态对话模板
messages = [
    {"role": "user", "image": [image],"content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}
]

# 应用模板并处理输入
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(model.device, torch.float16)

# 4. 生成回答
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id
    )

# 5. 解码输出
response = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

print(f"模型回答：{response}")