from typing import Optional, Tuple
from plotSig import *


# ==================== 自定义调制与滤波器函数 ====================
def rrcosfilter(N: int, alpha: float, Ts: float, Fs: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	生成根升余弦滤波器系数（与 MATLAB rcosine 等效）
	参数:
		N   : 滤波器阶数（输出长度 N+1）
		alpha: 滚降因子
		Ts  : 符号周期（秒）
		Fs  : 采样率（Hz）
	返回:
		t   : 时间轴
		h   : 滤波器系数（能量归一化）
	"""
	T_delta = 1.0 / Fs
	t = np.arange(-N // 2, N // 2 + 1) * T_delta
	h = np.zeros_like(t, dtype=float)
	
	for i, ti in enumerate(t):
		if ti == 0.0:
			h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
		elif abs(ti) == Ts / (4.0 * alpha):
			h[i] = (alpha / np.sqrt(2.0)) * (
					(1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
					(1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
			)
		else:
			num = (np.sin(np.pi * ti / Ts * (1.0 - alpha)) +
				   4.0 * alpha * ti / Ts * np.cos(np.pi * ti / Ts * (1.0 + alpha)))
			den = np.pi * ti / Ts * (1.0 - (4.0 * alpha * ti / Ts) ** 2)
			h[i] = num / den
	
	# 能量归一化
	h /= np.sqrt(np.sum(h ** 2))
	return t, h


def psk_mod(symbols: np.ndarray, M: int, phase_offset: float = 0.0) -> np.ndarray:
	"""M-PSK 调制，symbols 为 0..M-1 整数"""
	angles = 2.0 * np.pi * symbols / M + phase_offset
	return np.exp(1j * angles)


def qam_mod(symbols: np.ndarray, M: int) -> np.ndarray:
	"""方形 QAM 调制，symbols 为 0..M-1 整数"""
	sqrtM = int(np.sqrt(M))
	if sqrtM * sqrtM != M:
		raise ValueError("M must be a perfect square for QAM")
	real = (symbols // sqrtM) * 2 - (sqrtM - 1)
	imag = (symbols % sqrtM) * 2 - (sqrtM - 1)
	# 平均功率归一化
	return (real + 1j * imag) / np.sqrt((M - 1) * 2.0 / 3.0)


def msk_mod(bits: np.ndarray, samples_per_symbol: int) -> np.ndarray:
	"""
	MSK 调制，返回复基带信号（与 comm.MSKModulator 一致）
	采用差分编码 + 连续相位累加
	"""
	bits = np.asarray(bits, dtype=int)
	a = 2 * bits - 1  # 映射为 ±1
	theta = np.pi / (2.0 * samples_per_symbol)
	phase = 0.0
	out = []
	for ak in a:
		for _ in range(samples_per_symbol):
			out.append(np.exp(1j * phase))
			phase += theta * ak
	return np.array(out)


def gmsk_mod(bits: np.ndarray, samples_per_symbol: int, BT: float = 0.3, L: int = 1) -> np.ndarray:
	"""
	GMSK 调制，BT 为带宽时间积，L 为高斯滤波器长度（符号数）
	注：此简化实现使用高斯滤波器对矩形脉冲进行卷积
	"""
	bits = np.asarray(bits, dtype=int)
	a = 2 * bits - 1  # ±1
	
	# 生成高斯滤波器（截断到 L 个符号长度）
	Ts = 1.0  # 归一化符号周期
	Fs = samples_per_symbol  # 归一化采样率
	t_gauss = np.arange(-L * Fs, L * Fs + 1) / Fs
	sigma = np.sqrt(np.log(2)) / (2.0 * np.pi * BT)  # 3dB 带宽对应的 sigma
	gauss = np.exp(- (t_gauss ** 2) / (2.0 * sigma ** 2))
	gauss /= np.sum(gauss)  # 归一化
	
	# 对每个符号矩形脉冲进行高斯滤波，生成频率脉冲
	rect = np.ones(samples_per_symbol) / samples_per_symbol
	freq_pulse = np.convolve(rect, gauss, mode='full')
	freq_pulse = freq_pulse[: (2 * L + 1) * samples_per_symbol]
	
	# 相位累加
	phase = 0.0
	out = []
	# 将符号序列与频率脉冲卷积得到瞬时频率
	freq_signal = np.convolve(a, freq_pulse, mode='full')[: len(a) * samples_per_symbol]
	for f_inst in freq_signal:
		out.append(np.exp(1j * phase))
		phase += 2.0 * np.pi * f_inst / samples_per_symbol
	return np.array(out)


def oqpsk_mod(symbols: np.ndarray, samples_per_symbol: int, rc_alpha: float = 0.25) -> np.ndarray:
	"""
	OQPSK 调制，带根升余弦成型滤波
	"""
	# QPSK 映射 (0,1,2,3) -> 45°, 135°, 225°, 315°
	angles = np.pi / 4 + np.pi / 2 * symbols
	mod = np.exp(1j * angles)
	I = np.real(mod)
	Q = np.imag(mod)
	
	# 上采样并延迟 Q 路半个符号
	sps = samples_per_symbol
	I_up = np.zeros(len(I) * sps)
	I_up[::sps] = I
	Q_up = np.zeros(len(Q) * sps)
	Q_up[sps // 2::sps] = Q
	sig_up = I_up + 1j * Q_up
	
	# 根升余弦滤波器
	_, h_rrc = rrcosfilter(4 * sps, rc_alpha, 1.0, sps)
	filtered = np.convolve(sig_up, h_rrc, mode='same')
	return filtered


def upsample_and_filter(mod_unsampled: np.ndarray, sps: int, h_rrc: np.ndarray) -> np.ndarray:
	"""通用上采样与成型滤波"""
	upsampled = np.zeros(len(mod_unsampled) * sps, dtype=complex)
	upsampled[::sps] = mod_unsampled
	filtered = np.convolve(upsampled, h_rrc, mode='full')
	return filtered

# ==================== 主函数 ====================
def signal_generation(fb: float, fs: float, SNR: float, mod_type: str, time_sample: float) -> np.ndarray:
	"""
	生成指定调制类型的带噪基带信号。

	参数:
		fb         : 符号速率 (Hz)
		fs         : 采样率 (Hz)
		SNR        : 信噪比 (dB)
		mod_type   : 调制类型，支持 'msk','gmsk','oqpsk','bpsk','pi/4-bpsk','8psk','16qam','soqpsk'
		time_sample: 信号时长 (秒)

	返回:
		y          : 带噪复基带信号 (一维 numpy 数组)
	"""
	# 发射端过采样因子（固定为4倍符号速率）
	nsamp_tx = int(fs / fb)
	# 根升余弦滤波器（滚降系数0.25，群时延4符号）
	_, hcos_tx = rrcosfilter(4 * nsamp_tx, 0.25, 1.0 / fb, fb * nsamp_tx)
	
	# 采样点总数
	num_sample = int(np.floor(fs * time_sample))
	bit_num = num_sample*2
	
	# -------------------- 生成符号序列 --------------------
	if mod_type in ("oqpsk", "soqpsk"):
		x = np.random.randint(0, 4, bit_num)  # QPSK 符号
	elif mod_type == "8psk":
		x = np.random.randint(0, 8, bit_num)
	elif mod_type == "16qam":
		x = np.random.randint(0, 16, bit_num)
	else:
		x = np.random.randint(0, 2, bit_num)  # 比特
	
	# -------------------- 调制与成型滤波 --------------------
	if mod_type == "msk":
		modSignal = msk_mod(x, nsamp_tx)
	
	elif mod_type == "gmsk":
		modSignal = gmsk_mod(x, nsamp_tx, BT=0.3, L=1)
	
	elif mod_type == "oqpsk":
		modSignal_unsample = oqpsk_mod(x, nsamp_tx, rc_alpha=0.25)
		# 重采样至 fs
		modSignal = signal.resample(modSignal_unsample,
									int(len(modSignal_unsample) * fs / (fb * nsamp_tx)))
	
	elif mod_type == "bpsk":
		mod_unsampled = psk_mod(x, 2)
		mod_filtered = upsample_and_filter(mod_unsampled, nsamp_tx, hcos_tx)
		modSignal = signal.resample(mod_filtered, int(len(mod_filtered) * fs / (fb * nsamp_tx)))
	
	elif mod_type == "pi/4-bpsk":
		mod_unsampled = psk_mod(x, 2, phase_offset=np.pi / 4)
		mod_filtered = upsample_and_filter(mod_unsampled, nsamp_tx, hcos_tx)
		modSignal = signal.resample(mod_filtered, int(len(mod_filtered) * fs / (fb * nsamp_tx)))
	
	elif mod_type == "8psk":
		mod_unsampled = psk_mod(x, 8)
		mod_filtered = upsample_and_filter(mod_unsampled, nsamp_tx, hcos_tx)
		modSignal = signal.resample(mod_filtered, int(len(mod_filtered) * fs / (fb * nsamp_tx)))
	
	elif mod_type == "16qam":
		mod_unsampled = qam_mod(x, 16)
		mod_filtered = upsample_and_filter(mod_unsampled, nsamp_tx, hcos_tx)
		modSignal = signal.resample(mod_filtered, int(len(mod_filtered) * fs / (fb * nsamp_tx)))
	
	elif mod_type == "soqpsk":
		# QPSK 映射
		mod_qpsk = psk_mod(x, 4)
		I = np.real(mod_qpsk)
		Q = np.imag(mod_qpsk)
		I_up = np.zeros(len(I) * nsamp_tx)
		I_up[::nsamp_tx] = I
		Q_up = np.zeros(len(Q) * nsamp_tx)
		Q_up[::nsamp_tx] = Q
		# I/Q 延迟：I 路延迟半个符号，Q 路超前半个符号（与 MATLAB 代码一致）
		delay = nsamp_tx // 2
		I_up = np.concatenate([I_up, np.zeros(delay)])
		Q_up = np.concatenate([np.zeros(delay), Q_up])
		mod_upsample = I_up + 1j * Q_up
		mod_filtered = np.convolve(mod_upsample, hcos_tx, mode='full')
		modSignal = signal.resample(mod_filtered, int(len(mod_filtered) * fs / (fb * nsamp_tx)))
	else:
		raise ValueError(f"Unsupported modulation type: {mod_type}")
	
	# -------------------- 随机截取一段信号 --------------------
	if len(modSignal) <= num_sample:
		raise ValueError("Signal too short for requested time_sample. Increase fb or time_sample.")
	start_idx = np.random.randint(0, len(modSignal) - num_sample)
	SampleSignal = modSignal[start_idx: start_idx + num_sample]
	
	# -------------------- 添加 AWGN 噪声 --------------------
	pow_signal = np.sum(np.abs(SampleSignal) ** 2)
	snr_linear = 10 ** (SNR / 10.0)
	N0 = pow_signal / snr_linear / len(SampleSignal)  # 每采样点噪声功率
	sigma = np.sqrt(N0 / 2.0)
	noise = sigma * (np.random.randn(len(SampleSignal)) + 1j * np.random.randn(len(SampleSignal)))
	
	y = SampleSignal + noise
	return y


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ==================== 示例调用 ====================
if __name__ == "__main__":
	fb = 1e6
	fs = 192e6
	SNR = 15
	#mod_type:msk,gmsk,oqpsk,bpsk,pi/4-bpsk,8psk,16qam,soqpsk
	mod_type = "gmsk"
	num_fft = 4096
	time_sample = 22e-6
	
	y = signal_generation(fb, fs, SNR, mod_type, time_sample)
	print(f"生成信号长度: {len(y)} 采样点")
	print(f"信号平均功率: {np.mean(np.abs(y) ** 2):.4f}")
	
	plot_time_domain(y, fs, xlim=(0, time_sample/2))
	
	plot_spectrogram(y, fs, num_fft=num_fft, nperseg=256, noverlap=200,
					 freq_range=(-10e6, 10e6), title=f"{mod_type} Spectrogram")
	
	# 显示所有采样点（噪声下的轨迹）
	# plot_constellation(y, show_all_samples=True, alpha=0.3, title=f"{mod_type} Trajectory (All Samples)")
	
	# 仅显示符号点（星座图）
	# plot_constellation(y, fs=fs, fb=fb, show_all_samples=False, title=f"{mod_type} Symbol Constellation")
	
	plot_constellation2(y)
	plt.show()
	