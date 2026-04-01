import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/home/shared/models/Qwen/Qwen3___5-9B"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

prompt = "你支持图片输入吗。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(response)