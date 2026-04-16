close all;
fb = 1e6;
fs = 192e6;
SNR = 20;
% mod_type:msk,gmsk,oqpsk,bpsk,pi/4-bpsk,8psk,16qam,soqpsk;
mod_type = "gmsk";
time_sample = 100e-6;
num_fft = 4096;

%生成复数信号
y = signal_generation(fb, fs, SNR, mod_type, time_sample);

% 绘制 IQ 时序图
plot_iq_timing(y, fs, [0, time_sample/2]);

% 绘制时频图，观察 -10 MHz ~ 10 MHz 范围
plot_time_freq(y, fs, num_fft, [-5e6, 5e6]);

% 绘制星座图
plot_constellation(y);

function plot_iq_timing(y, fs, time_lim)
% plot_iq_timing 绘制复数信号的 I/Q 时序图
%   y        - 复数信号向量
%   fs       - 采样率 (Hz)
%   time_lim - [可选] 时间轴显示范围，格式 [t_start, t_end] (秒)
%              若不提供，默认显示整个信号时长。

    if nargin < 3
        time_lim = [0, (length(y)-1)/fs];
    end

    t = (0:length(y)-1) / fs;          % 时间轴
    I = real(y);
    Q = imag(y);

    figure;
    plot(t, I, 'b-', 'LineWidth', 1.2); hold on;
    plot(t, Q, 'r-', 'LineWidth', 1.2);
    grid on;
    xlim(time_lim);
    xlabel('时间 (s)');
    ylabel('幅度');
    legend('I', 'Q');
    title('IQ 时序图');
end

function plot_time_freq(y, fs, num_fft, freq_range)
% plot_time_freq 绘制复数信号的时频图（频谱图）
%   y          - 复数信号向量
%   fs         - 采样率 (Hz)
%   num_fft    - FFT 点数，决定频率分辨率
%   freq_range - 频率显示范围，格式 [f_low, f_high] (Hz)

    if nargin < 4
        freq_range = [-fs/2, fs/2];     % 默认显示全频带
    end

    % 计算频谱图，返回功率谱密度 (dB)
    [S, F, T] = spectrogram(y, hamming(num_fft), num_fft/2, num_fft, fs, 'centered', 'psd');

    % 功率转 dB
    S_dB = 10*log10(abs(S) + eps);

    figure;
    imagesc(T, F, S_dB);
    axis xy;
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
    ylim(freq_range);
    colormap('jet');
    colorbar;
    title(sprintf('时频图 (FFT点数 = %d)', num_fft));
end

function plot_constellation(y)
% plot_constellation 绘制复数信号的星座图
%   y - 复数信号向量

    figure;
    plot(real(y), imag(y), 'b.', 'MarkerSize', 4);
    grid on;
    axis equal;
    xlabel('同相分量 (I)');
    ylabel('正交分量 (Q)');
    title('星座图');
end
