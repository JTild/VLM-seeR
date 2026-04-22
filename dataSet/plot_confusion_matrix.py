import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========== 配置 ==========
PRED_FILE = "/home/jql/code/LLaMA-Factory/saves/Qwen2.5-VL-72B-Instruct/lora/eval_2026-04-20-10-32-35/generated_predictions.jsonl"
DATASET_ROOT = "./sig_dataset417v2_test"
OUTPUT_IMAGE = DATASET_ROOT+"/confusion_matrix.png"

def clean_label(text):
	"""清洗标签：去空格、统一小写"""
	return text.strip().lower()


def load_predictions(file_path):
	y_true = []
	y_pred = []
	with open(file_path, "r", encoding="utf-8") as f:
		for line in f:
			data = json.loads(line)
			y_true.append(clean_label(data["label"]))
			y_pred.append(clean_label(data["predict"]))
	return y_true, y_pred


def main():
	y_true, y_pred = load_predictions(PRED_FILE)
	
	# 获取清洗后的唯一类别
	labels = sorted(list(set(y_true + y_pred)))
	print("检测到的类别：", labels)
	
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	fig, ax = plt.subplots(figsize=(10, 8))
	disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
	plt.title("Confusion Matrix (Cleaned Labels)")
	plt.tight_layout()
	plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
	print(f"混淆矩阵已保存为 {OUTPUT_IMAGE}")


if __name__ == "__main__":
	main()