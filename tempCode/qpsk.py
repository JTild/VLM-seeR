import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# ================= 参数配置 =================
fs = 10e6               # 采样率 10 MHz
Rs = 1e6                # 符号率 1 MHz
sps = int(fs / Rs)      # 每个符号的采样点数 = 10
num_symbols = 200       # 发送符号个数
alpha = 0.35            # 升余弦滚降系数
filter_len = 64         # 成形滤波器抽头数（奇数）

fft_len = 512           # STFT 的 FFT 点数
fc = 2e6                # 可选：中频载波频率 (2 MHz)

# ================= 生成随机比特 =================
bits = np.random.randint(0, 2, num_symbols * 2)   # 总比特数 = 符号数 * 2
# 串并转换，每两个比特组成一个符号
bits_I = bits[0::2]      # 偶数索引 -> I 路
bits_Q = bits[1::2]      # 奇数索引 -> Q 路

# ================= QPSK 符号映射 (Gray 映射) =================
# 00 -> 1+j, 01 -> -1+j, 11 -> -1-j, 10 -> 1-j
def map_qpsk(bits_I, bits_Q):
    # 映射 I 路: 0->1, 1->-1
    I = np.where(bits_I == 0, 1, -1)
    Q = np.where(bits_Q == 0, 1, -1)
    return I + 1j * Q

symbols = map_qpsk(bits_I, bits_Q)   # 复数符号序列，长度 = num_symbols

# ================= 上采样（插入零） =================
# 在每个符号之间插入 sps-1 个零
upsampled = np.zeros(num_symbols * sps, dtype=complex)
upsampled[::sps] = symbols

# ================= 升余弦滚降成形滤波器 =================
# 设计平方根升余弦滤波器（发射端使用 RRC，接收端再匹配；这里直接使用 RC 作为成形）
# 为了简化，使用普通升余弦滤波器（RC），保证无 ISI
t_filter = np.arange(-filter_len//2, filter_len//2 + 1) / Rs
h_rc = np.sinc(t_filter * Rs) * np.cos(np.pi * alpha * t_filter * Rs) / (1 - (2 * alpha * t_filter * Rs) ** 2)
# 处理除零情况
h_rc[np.isnan(h_rc)] = 1.0   # 中心点修正
h_rc /= np.sum(h_rc)         # 归一化

# 对 upsampled 序列进行滤波（卷积）
tx_baseband = lfilter(h_rc, 1.0, upsampled)

# 可选：将基带信号搬到载波 fc 上，得到带通信号（便于观察时频图中心频率非零）
t = np.arange(len(tx_baseband)) / fs
tx_passband = tx_baseband * np.exp(1j * 2 * np.pi * fc * t)

# 选择绘制带通信号（也可绘制基带信号）
signal_to_plot = tx_baseband   # 复数带通信号

# ================= 绘图 =================
plt.figure(figsize=(14, 10))

# 1) 时域 I/Q 波形（前 512 个采样点）
plt.subplot(3, 1, 1)
plot_len = min(512, len(signal_to_plot))
t_plot = t[:plot_len] * 1e6      # 微秒
plt.plot(t_plot, np.real(signal_to_plot[:plot_len]), label='I (real)', linewidth=0.8)
plt.plot(t_plot, np.imag(signal_to_plot[:plot_len]), label='Q (imag)', linewidth=0.8, alpha=0.7)
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.title(f'QPSK Modulated Signal (First {plot_len} Samples, Rs={Rs/1e6} MHz)')
plt.legend()
plt.grid(True)

# 2) 星座图（发射符号）
plt.subplot(3, 1, 2)
plt.scatter(np.real(symbols), np.imag(symbols), s=20, alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('In-phase')
plt.ylabel('Quadrature')
plt.title('QPSK Constellation (Ideal)')
plt.grid(True)
plt.axis('equal')

# 3) 时频图（STFT）
plt.subplot(3, 1, 3)
# 使用 plt.specgram 绘制复数信号的谱图
Pxx, freqs, bins, im = plt.specgram(
    signal_to_plot,
    NFFT=fft_len,
    Fs=fs,
    noverlap=fft_len - 32,      # 重叠 480 点，提高时间分辨率
    cmap='viridis',
    mode='psd'                  # 功率谱密度 (dB/Hz)
)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title(f'QPSK Spectrogram (fs={fs/1e6} MHz, Rs={Rs/1e6} MHz, NFFT={fft_len})')
plt.colorbar(label='Power Spectral Density (dB/Hz)')

plt.tight_layout()
plt.show()

# 打印一些信息
print(f"采样率: {fs/1e6} MHz")
print(f"符号率: {Rs/1e6} MHz")
print(f"每个符号采样点数: {sps}")
print(f"滚降系数: {alpha}")
print(f"总采样点数: {len(tx_baseband)}")
print(f"信号持续时间: {len(tx_baseband)/fs*1000:.2f} ms")