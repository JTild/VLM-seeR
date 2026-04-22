warning off;
% 设置参数
params.fb = 1e6;
params.fs = 192e6;
params.SNR = [10, 20];          % SNR 在 10~20 dB 均匀随机
% params.SNR = 15;
params.time_sample = 100e-6;
params.num_fft = 4096;
params.root_dir = './sig_dataset417v2_test'; 
% mod_type:msk,gmsk,oqpsk,bpsk,pi/4-bpsk,8psk,16qam,soqpsk;
mod_types = ["msk","gmsk","oqpsk","bpsk","pi/4-bpsk","8psk","16qam","soqpsk"];
num_per_class = 15;

generate_signal_dataset_multiThread(mod_types, num_per_class, params);

function generate_signal_dataset_multiThread(mod_types, num_per_class, params)

    % 检查并行池，若未启动则启动（指定进程数）
    pool = gcp('nocreate');
    if isempty(pool)
        % 根据你的核心数调整
        parpool('local', 64);
    else
        fprintf('使用已有并行池，Worker 数量: %d\n', pool.NumWorkers);
    end

    % 解析参数
    fb = params.fb;
    fs = params.fs;
    SNR_spec = params.SNR;
    time_sample = params.time_sample;
    num_fft = params.num_fft;
    root_dir = params.root_dir;

    % 创建目录（主进程执行）
    iq_dir = fullfile(root_dir, 'iq_timing');
    tf_dir = fullfile(root_dir, 'time_freq');
    const_dir = fullfile(root_dir, 'constellation');
    sig_dir = fullfile(root_dir, 'signals');
    dirs = {iq_dir, tf_dir, const_dir, sig_dir};
    for d = 1:length(dirs)
        if ~exist(dirs{d}, 'dir')
            mkdir(dirs{d});
        end
    end

    % 判断 SNR 是固定值还是范围
    if numel(SNR_spec) == 1
        SNR_is_range = false;
    else
        SNR_is_range = true;
        SNR_min = SNR_spec(1);
        SNR_max = SNR_spec(2);
    end

    % 预先计算总样本数及各调制类型的起始索引，用于确定文件名中的全局编号
    total_samples = length(mod_types) * num_per_class;
    mod_index_start = 1;
    mod_offset = zeros(length(mod_types), 1);
    for m = 1:length(mod_types)
        mod_offset(m) = mod_index_start;
        mod_index_start = mod_index_start + num_per_class;
    end

    % 预分配元数据单元格数组（每个 worker 返回其处理部分的元数据）
    metadata_cells = cell(length(mod_types), num_per_class);

    % 并行循环：外层为调制类型，内层为样本（合并为一个并行循环）
    % 为了便于分配全局索引，使用线性索引
    all_tasks = [];
    for m = 1:length(mod_types)
        for n = 1:num_per_class
            task.mod_idx = m;
            task.sample_in_class = n;
            all_tasks = [all_tasks; task];
        end
    end

    % 并行执行所有任务
    parfor task_id = 1:length(all_tasks)
        warning off;
        m = all_tasks(task_id).mod_idx;
        n = all_tasks(task_id).sample_in_class;
        mod_type = mod_types(m);

        % 确定 SNR 值
        if SNR_is_range
            SNR_val = 10^( log10(SNR_min) + (log10(SNR_max)-log10(SNR_min)) * rand() );
        else
            SNR_val = SNR_spec;
        end
        % 生成复数信号
        y = signal_generation(fb, fs, SNR_val, mod_type, time_sample);

        % 构造文件名中的全局编号
        global_idx = mod_offset(m) + (n-1);
        mod_type_clean = regexprep(char(mod_type), '[\\/:*?"<>|]', '_');
        file_id = sprintf('sig_%06d_%s_snr%.1f', global_idx, mod_type_clean, SNR_val);
        mat_file = [file_id, '.mat'];
        png_iq   = [file_id, '_iq.png'];
        png_tf   = [file_id, '_tf.png'];
        png_const= [file_id, '_const.png'];

        % 保存复数信号数据（每个 worker 独立写文件，无冲突）
        data_to_save = struct('y', y, 'fb', fb, 'fs', fs, ...
                              'SNR_val', SNR_val, 'mod_type', mod_type, ...
                              'time_sample', time_sample);
        par_save(fullfile(sig_dir, mat_file), data_to_save);

        % 生成并保存 IQ 时序图
        fig_iq = figure('Visible', 'off');
        plot_iq_timing(y, fs, [0, time_sample/2]);
        par_saveas(fig_iq, fullfile(iq_dir, png_iq));
        close(fig_iq);

        % 生成并保存时频图
        fig_tf = figure('Visible', 'off');
        plot_time_freq(y, fs, num_fft, [-5e6, 5e6]);
        par_saveas(fig_tf, fullfile(tf_dir, png_tf));
        close(fig_tf);

        % 生成并保存星座图
        fig_const = figure('Visible', 'off');
        plot_constellation(y);
        par_saveas(fig_const, fullfile(const_dir, png_const));
        close(fig_const);

        % 组装元数据行（作为结构体返回）
        metadata_cells{task_id} = struct(...
            'signal_file', string(mat_file), ...
            'mod_type', string(mod_type), ...
            'SNR_dB', SNR_val, ...
            'iq_image', string(png_iq), ...
            'tf_image', string(png_tf), ...
            'const_image', string(png_const));
    end

    % 汇总元数据（主进程）
    metadata_table = table();
    for k = 1:numel(metadata_cells)
        if ~isempty(metadata_cells{k})
            metadata_table = [metadata_table; struct2table(metadata_cells{k})];
        end
    end

    % 写入 CSV
    writetable(metadata_table, fullfile(root_dir, 'metadata.csv'));
    fprintf('数据集生成完成！共 %d 个样本，保存于 %s\n', height(metadata_table), root_dir);
    % 获取当前并行池（不新建）
    pool = gcp('nocreate');

    % 如果存在，就关闭
    if ~isempty(pool)
        delete(pool);
    end
end

% ------------------------------------------------------------------------
% 辅助函数：parfor 中保存 .mat 文件（避免透明度问题）
function par_save(filename, data_struct)
    save(filename, '-struct', 'data_struct', '-v7');
end

% 辅助函数：parfor 中保存图形（确保 saveas 在 worker 上正常工作）
function par_saveas(fig_handle, filename)
    saveas(fig_handle, filename);
end

% ------------------------------------------------------------------------
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