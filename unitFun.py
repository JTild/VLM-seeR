import numpy as np

def requiredSignalLengthForSquareSpectrogram(fs, L, overlapRatio=0.5):
    """
    计算使 STFT 时频图为方形所需的最小信号长度。

    参数：
        fs : float
            采样率 (Hz)
        L : int
            FFT 点数（窗长），通常为偶数
        overlapRatio : float, 默认 0.5
            重叠率，取值 [0, 1)。步长 hop = L * (1 - overlapRatio)

    返回：
        N : int
            所需的采样点数
        T : float
            对应的信号时长 (秒)
        M : int
            方形时频图的边长（时间点数 = 频率点数）
    """
    hop = int(L * (1 - overlapRatio))
    if hop <= 0:
        raise ValueError("重叠率过大，步长必须大于 0。")

    # 频率点数（实信号单边谱）
    M = L // 2 + 1

    # 理想情况：(N - L)/hop + 1 = M  => N = (M - 1) * hop + L
    N_ideal = (M - 1) * hop + L

    # 验证 floor 后是否严格等于 M，微调 N
    M_t_calc = (N_ideal - L) // hop + 1
    if M_t_calc != M:
        while M_t_calc < M:
            N_ideal += 1
            M_t_calc = (N_ideal - L) // hop + 1
        while M_t_calc > M:
            N_ideal -= 1
            M_t_calc = (N_ideal - L) // hop + 1

    N = N_ideal
    T = N / fs
    return N, T, M

# ========== 示例用法 ==========
if __name__ == "__main__":
    fs = 40e6          # 40 MHz
    L = 1024
    overlap = 0.5

    N, T, M = requiredSignalLengthForSquareSpectrogram(fs, L, overlap)

    print(f"采样率 fs = {fs/1e6:.1f} MHz")
    print(f"FFT 点数 L = {L}")
    print(f"重叠率 = {overlap*100:.0f}% (hop = {int(L*(1-overlap))})")
    print(f"所需的采样点数 N = {N}")
    print(f"对应的信号时长 T = {T*1000:.6f} ms")
    print(f"方形时频图尺寸 = {M} × {M}")
    print(f"验证：时间点数 = {(N-L)//(L//2)+1}，频率点数 = {L//2+1}")