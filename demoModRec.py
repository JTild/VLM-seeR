from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)

modelPath = "/home/jql/code/Qwen2.5vl/Qlorav1"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    modelPath, torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(modelPath)

image_paths = [
	"./dataSet/sig_dataset/iq_timing/sig_000001_msk_snr15.0_iq.png",
	"./dataSet/sig_dataset/time_freq/sig_000001_msk_snr15.0_tf.png",
	"./dataSet/sig_dataset/constellation/sig_000001_msk_snr15.0_const.png",
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
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
print(f"输入 token 数量（占位符阶段）: {inputs.input_ids.shape[1]}")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
