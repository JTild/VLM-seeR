import matplotlib.pyplot as plt
import numpy as np
from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.utils.data_loading import WorkerSeedingDataLoader

# 加载训练集静态数据（注意 target_labels 包含 class_index 和可选的 class_name）
train_dataset = StaticTorchSigDataset(
    root="./classifier_example/train",
    target_labels=["class_index"],   # 也可添加 "class_name"，但需要生成时保存
)

# 获取类别名称列表（从数据集的属性中获取）
class_names =  TorchSigSignalLists.all_signals  # 或者 train_dataset.class_names

# 取第一个样本（索引0）
sample_data, label_idx = train_dataset[0]

# 获取调制类型名称
modulation_type = class_names[label_idx]
print(f"样本调制类型: {modulation_type} (索引 {label_idx})")

# 数据形状: (2, 65536) -> 第一通道实部，第二通道虚部
real_part = sample_data[0]
imag_part = sample_data[1]
complex_iq = real_part + 1j * imag_part   # 恢复复数IQ信号

# 1. 绘制时序图（I/Q 分量）
time_axis = np.arange(len(complex_iq))
plt.figure(figsize=(12, 5))
plt.plot(time_axis, real_part, label='I ', alpha=0.7)
plt.plot(time_axis, imag_part, label='Q ', alpha=0.7)
plt.xlabel('samplePoint')
plt.ylabel('AM')
plt.title(f'timeFreq - {modulation_type}')
plt.legend()
plt.grid(True)

# 2. 绘制时频图（短时傅里叶变换谱图）
plt.figure(figsize=(12, 5))
Pxx, freqs, bins, im = plt.specgram(complex_iq, NFFT=256, Fs=10000, noverlap=128,
                                    cmap='viridis', mode='magnitude')
plt.xlabel('time (S)')
plt.ylabel('freq(Hz)')
plt.title(f'time-freQ (STFT) - {modulation_type}')
plt.colorbar(label='aM (dB)')
plt.tight_layout()
plt.show()