warning off;
% 设置参数
params.fb = 1e6;
params.fs = 192e6;
% params.SNR = [10, 20];          % SNR 在 10~20 dB 均匀随机
params.SNR = 15;
params.time_sample = 100e-6;
params.num_fft = 4096;
params.root_dir = './sig_dataset';
% mod_type:msk,gmsk,oqpsk,bpsk,pi/4-bpsk,8psk,16qam,soqpsk;
mod_types = ["msk","gmsk","oqpsk","bpsk","pi/4-bpsk","8psk","16qam","soqpsk"];
num_per_class = 1;

generate_signal_dataset(mod_types, num_per_class, params);

function generate_signal_dataset(mod_types, num_per_class, params)
%   mod_types      - 字符串数组，如 ["gmsk","bpsk","qpsk"]
%   num_per_class  - 每类调制方式生成的样本数
%   params         - 结构体，包含 fb, fs, SNR, time_sample, num_fft, root_dir
%
% 示例：
%   params.fb = 1e6;
%   params.fs = 192e6;
%   params.SNR = [10, 20];          % 或固定值 15
%   params.time_sample = 100e-6;
%   params.num_fft = 4096;
%   params.root_dir = './sig_dataset';
%   generate_signal_dataset(["gmsk","bpsk","qpsk"], 100, params);

    % 解析参数
    fb = params.fb;
    fs = params.fs;
    SNR_spec = params.SNR;
    time_sample = params.time_sample;
    num_fft = params.num_fft;
    root_dir = params.root_dir;

    % 创建目录
    iq_dir = fullfile(root_dir, 'iq_timing');
    tf_dir = fullfile(root_dir, 'time_freq');
    const_dir = fullfile(root_dir, 'constellation');
    sig_dir = fullfile(root_dir, 'signals');
    if ~exist(root_dir, 'dir'), mkdir(root_dir); end
    if ~exist(iq_dir, 'dir'), mkdir(iq_dir); end
    if ~exist(tf_dir, 'dir'), mkdir(tf_dir); end
    if ~exist(const_dir, 'dir'), mkdir(const_dir); end
    if ~exist(sig_dir, 'dir'), mkdir(sig_dir); end

    % 准备元数据表
    metadata = table();
    sample_idx = 0;

    % 判断 SNR 是固定值还是范围
    if numel(SNR_spec) == 1
        SNR_is_range = false;
    else
        SNR_is_range = true;
        SNR_min = SNR_spec(1);
        SNR_max = SNR_spec(2);
    end

    % 循环生成
    for m = 1:length(mod_types)
        mod_type = mod_types(m);
        fprintf('生成调制类型: %s\n', mod_type);

        for n = 1:num_per_class
            sample_idx = sample_idx + 1;

            % 确定 SNR 值
            if SNR_is_range
                SNR_val = SNR_min + (SNR_max - SNR_min) * rand();
            else
                SNR_val = SNR_spec;
            end

            % 生成复数信号
            y = signal_generation(fb, fs, SNR_val, mod_type, time_sample);

            % 构造文件名（不含扩展名）
            mod_type_clean = regexprep(char(mod_type), '[\\/:*?"<>|]', '_');
            file_id = sprintf('sig_%06d_%s_snr%.1f', sample_idx, mod_type_clean, SNR_val);
            mat_file = [file_id, '.mat'];
            png_iq   = [file_id, '_iq.png'];
            png_tf   = [file_id, '_tf.png'];
            png_const= [file_id, '_const.png'];

            % 保存复数信号数据
            save(fullfile(sig_dir, mat_file), 'y', 'fb', 'fs', 'SNR_val', 'mod_type', 'time_sample','-v7');

            % 生成并保存 IQ 时序图
            fig_iq = figure('Visible', 'off');
            plot_iq_timing(y, fs, [0, time_sample/2]);
            saveas(fig_iq, fullfile(iq_dir, png_iq));
            close(fig_iq);

            % 生成并保存时频图
            fig_tf = figure('Visible', 'off');
            plot_time_freq(y, fs, num_fft, [-5e6, 5e6]);   % 频宽可按需修改
            saveas(fig_tf, fullfile(tf_dir, png_tf));
            close(fig_tf);

            % 生成并保存星座图
            fig_const = figure('Visible', 'off');
            plot_constellation(y);
            saveas(fig_const, fullfile(const_dir, png_const));
            close(fig_const);

            % 追加元数据
            new_row = table({mat_file}, {char(mod_type)}, SNR_val, ...
                            {png_iq}, {png_tf}, {png_const}, ...
                            'VariableNames', {'signal_file','mod_type','SNR_dB', ...
                                              'iq_image','tf_image','const_image'});
            metadata = [metadata; new_row];
        end
    end

    % 写入元数据 CSV
    writetable(metadata, fullfile(root_dir, 'metadata.csv'));
    fprintf('数据集生成完成！共 %d 个样本，保存于 %s\n', sample_idx, root_dir);
end

function plot_iq_timing(y, fs, time_lim)
    if nargin < 3
        time_lim = [0, (length(y)-1)/fs];
    end
    t = (0:length(y)-1) / fs;
    I = real(y);
    Q = imag(y);
    plot(t, I, 'b-', 'LineWidth', 1.2); hold on;
    plot(t, Q, 'r-', 'LineWidth', 1.2);
    grid on;
    xlim(time_lim);
%     xlabel('时间 (s)');
%     ylabel('幅度');
%     legend('I', 'Q');
%     title('IQ 时序图');
end

function plot_time_freq(y, fs, num_fft, freq_range)
    if nargin < 4
        freq_range = [-fs/2, fs/2];
    end
    [S, F, T] = spectrogram(y, hamming(num_fft), num_fft/2, num_fft, fs, 'centered', 'psd');
    S_dB = 10*log10(abs(S) + eps);
    imagesc(T, F, S_dB);
    axis xy;
%     xlabel('时间 (s)');
%     ylabel('频率 (Hz)');
    ylim(freq_range);
    colormap('jet');
%     colorbar;
%     title(sprintf('时频图 (FFT点数 = %d)', num_fft));
end

function plot_constellation(y)
    plot(real(y), imag(y), 'b.', 'MarkerSize', 4);
    grid on;
    axis equal;
%     xlabel('同相分量 (I)');
%     ylabel('正交分量 (Q)');
%     title('星座图');
end