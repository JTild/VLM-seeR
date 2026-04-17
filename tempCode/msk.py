import numpy as np
import matplotlib.pyplot as plt

# ================= 参数配置 =================
fs = 10e6               # 采样率 10 MHz
symbol_rate = 100e3     # 符号率 100 kbps（每比特周期 Tb = 1/symbol_rate）
samples_per_symbol = int(fs / symbol_rate)  # 每个符号的采样点数 = 100

num_symbols = 200       # 调制符号个数（即比特个数）
fft_len = 512           # STFT 的 FFT 点数（用户要求）

# 频偏 Δf = 1/(4*Tb) = symbol_rate / 4
delta_f = symbol_rate / 4   # = 25 kHz

# ================= 生成随机比特 =================
bits = np.random.randint(0, 2, num_symbols)   # 0/1 序列
# 映射为 ±1：1 -> +1, 0 -> -1
a = 2 * bits - 1          # shape: (num_symbols,)

# ================= MSK 基带调制 =================
# 初始化相位累积变量
phase = 0.0
phase_acc = []            # 存储每个采样点的瞬时相位

# 对每个符号，生成 samples_per_symbol 个采样点的线性相位变化
for k in range(num_symbols):
    # 当前符号的相位变化斜率： (π * a_k) / (2 * Tb) 弧度/秒
    # 等效为每个采样点增加的相位增量： (π * a_k) / (2 * Tb) * (1/fs)
    phase_increment = np.pi * a[k] / (2 * symbol_rate) * (1 / fs)   # 每采样点弧度
    # 或者直接使用频偏： phase_increment = 2 * np.pi * (a[k] * delta_f) / fs
    # 两种等价，这里采用相位斜率法

    for _ in range(samples_per_symbol):
        phase += phase_increment
        phase_acc.append(phase)

phase_acc = np.array(phase_acc)   # 总长度 = num_symbols * samples_per_symbol

# 复基带信号：s(t) = exp(j * phase(t))
msk_baseband = np.exp(1j * phase_acc)

# 可选：为展示时频图更直观，可以将基带信号搬到载波 fc = 2 MHz 上（非必须）
fc = 2e6                 # 中频载波 2 MHz
t = np.arange(len(msk_baseband)) / fs
msk_passband = msk_baseband * np.exp(1j * 2 * np.pi * fc * t)

# 我们选择绘制带通信号的时频图（中心频率 fc = 2 MHz），也可直接绘制基带信号
signal_to_plot = msk_passband   # 复数带通信号

# ================= 绘图 =================
plt.figure(figsize=(14, 6))

# 1) 时域（前 512 个采样点）
plt.subplot(2, 1, 1)
plot_len = min(512, len(signal_to_plot))
t_plot = t[:plot_len] * 1e6      # 转为微秒
plt.plot(t_plot, np.real(signal_to_plot[:plot_len]), label='I (real)')
plt.plot(t_plot, np.imag(signal_to_plot[:plot_len]), label='Q (imag)', alpha=0.7)
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.title('MSK Modulated Signal (First 512 Samples)')
plt.legend()
plt.grid(True)

# 2) 时频图（STFT）
plt.subplot(2, 1, 2)
# 使用 plt.specgram 绘制复数信号的谱图
# Fs = fs, NFFT = fft_len (用户要求512)
Pxx, freqs, bins, im = plt.specgram(
    signal_to_plot,
    NFFT=fft_len,
    Fs=fs,
    noverlap=fft_len - 32,   # 重叠512-32=480，提高时间分辨率
    cmap='viridis',
    mode='psd'               # 功率谱密度 (dB)
)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'MSK Spectrogram (Fs={fs/1e6} MHz, NFFT={fft_len})')
plt.colorbar(label='Power Spectral Density (dB/Hz)')
plt.tight_layout()
plt.show()

# 打印一些基本统计信息
print(f"采样率: {fs/1e6} MHz")
print(f"符号率: {symbol_rate/1e3} kbps")
print(f"频偏 Δf: {delta_f/1e3} kHz")
print(f"总采样点数: {len(signal_to_plot)}")
print(f"持续时间: {len(signal_to_plot)/fs*1000:.2f} ms")