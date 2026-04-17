import matplotlib.pyplot as plt
import numpy as np


def plot_time_domain(y: np.ndarray, fs: float, title: str = "Signal Time Domain",
					 xlim: tuple = None, ylim: tuple = None,
					 show_iq: bool = True, i_label: str = "I (Real)", q_label: str = "Q (Imag)"):
	"""
	绘制复基带信号的时序图（实部和虚部）。

	参数:
		y        : 复信号数组
		fs       : 采样率 (Hz)
		title    : 图标题
		xlim     : (xmin, xmax) 时间轴范围（秒），None 则自动
		ylim     : (ymin, ymax) 幅度轴范围，None 则自动
		show_iq  : 是否同时显示 I/Q 两路（True）或仅显示实部（False）
		i_label  : I 路图例标签
		q_label  : Q 路图例标签
	"""
	t = np.arange(len(y)) / fs
	plt.figure(figsize=(10, 4))
	
	if show_iq:
		plt.plot(t, np.real(y), linewidth=0.8, label=i_label)
		plt.plot(t, np.imag(y), linewidth=0.8, linestyle='--', label=q_label)
		plt.legend(loc='best')
		plt.ylabel("Amplitude")
	else:
		plt.plot(t, np.real(y), linewidth=0.8)
		plt.ylabel("Amplitude (Real Part)")
	
	plt.xlabel("Time (s)")
	# plt.title(title)
	plt.grid(True, alpha=0.3)
	
	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim)
	
	plt.tight_layout()


def plot_spectrogram(y: np.ndarray, fs: float, num_fft: int = 4096,
					 nperseg: int = 256, noverlap: int = 192,
					 freq_range: tuple = None, title: str = "Spectrogram",
					 cmap: str = "viridis"):
	"""
	绘制信号的时频图（频谱图），横轴时间，纵轴频率，颜色表示功率谱密度(dB)。

	参数:
		y          : 复信号数组
		fs         : 采样率 (Hz)
		num_fft    : FFT 点数（频率分辨率）
		nperseg    : 每段长度（窗口大小）
		noverlap   : 重叠点数
		freq_range : (fmin, fmax) 频率显示范围 (Hz)，None 则显示全范围
		title      : 图标题
		cmap       : 颜色映射
	"""
	# 计算频谱图
	f, t_spec, Sxx = signal.spectrogram(y, fs, nperseg=nperseg, noverlap=noverlap,
										nfft=num_fft, return_onesided=False)
	# 对于复基带信号，通常显示双边谱，将零频移至中心
	f_shifted = np.fft.fftshift(f)
	Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
	Sxx_db = 20 * np.log10(np.abs(Sxx_shifted) + 1e-12)  # 防止 log(0)
	
	plt.figure(figsize=(10, 6))
	plt.pcolormesh(t_spec, f_shifted / 1e6, Sxx_db, shading='auto', cmap=cmap)
	plt.xlabel("Time (s)")
	plt.ylabel("Frequency (MHz)")
	# plt.title(title)
	plt.colorbar(label="Power/Frequency (dB)")
	
	# 限制频率显示范围
	if freq_range:
		plt.ylim(freq_range[0] / 1e6, freq_range[1] / 1e6)  # 转换为 MHz
	plt.tight_layout()


# plt.show()

def plot_constellation(y: np.ndarray, fs: float = None, fb: float = None,
					   sps: int = None, title: str = "Constellation Diagram",
					   xlim: tuple = (-1 / 4, 1 / 4), ylim: tuple = (-1 / 4, 1 / 4),
					   alpha: float = 0.5, marker_size: float = 2,
					   show_all_samples: bool = False):
	"""
	绘制复基带信号的星座图。

	参数:
		y                : 复信号数组
		fs               : 采样率 (Hz)，若 show_all_samples=False 且 sps 未提供时需与 fb 一起使用
		fb               : 符号速率 (Hz)，用于计算最佳采样点
		sps              : 每个符号的采样点数 (samples per symbol)，可替代 fs/fb
		title            : 图标题
		xlim, ylim       : 坐标轴范围
		alpha            : 散点透明度 (0~1)
		marker_size      : 散点大小
		show_all_samples : 是否显示所有采样点，若为 False 则仅显示符号点（最佳采样时刻）
	"""
	plt.figure(figsize=(6, 6))
	
	if show_all_samples:
		# 显示所有采样点
		plt.scatter(np.real(y), np.imag(y), s=marker_size, alpha=alpha, edgecolors='none')
	else:
		# 仅显示符号点（每个符号的最佳采样时刻，即每个符号周期中间或末尾）
		if sps is None:
			if fs is None or fb is None:
				raise ValueError("Either provide 'sps' or both 'fs' and 'fb' for symbol point extraction.")
			sps = int(round(fs / fb))
		# 取每个符号的中间点（或末尾点，根据常用习惯）
		idx = np.arange(sps // 2, len(y), sps)  # 假设眼图张开最大时刻在符号中点
		y_symbols = y[idx]
		plt.scatter(np.real(y_symbols), np.imag(y_symbols),
					s=marker_size * 3, alpha=alpha, edgecolors='none', c='blue')
	
	plt.xlabel("In-phase (I)")
	plt.ylabel("Quadrature (Q)")
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
	plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
	plt.axis('equal')
	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim)
	plt.tight_layout()


# plt.show()


def plot_constellation2(signal):
	"""
	绘制复数信号的星座图（基础版）。

	Parameters
	----------
	signal : array_like of complex
		复数符号序列。
	"""
	signal = np.asarray(signal).flatten()
	if not np.iscomplexobj(signal):
		raise ValueError("输入信号必须为复数数组。")
	
	I = signal.real
	Q = signal.imag
	
	plt.figure(figsize=(6, 6))
	plt.scatter(I, Q, s=5, c='blue', alpha=0.6, edgecolors='none')
	plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
	plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
	plt.xlabel('In-Phase (I)')
	plt.ylabel('Quadrature (Q)')
	plt.title('Constellation Diagram')
	plt.grid(True, linestyle=':', alpha=0.6)
	plt.axis('equal')
	plt.tight_layout()
	plt.show()
