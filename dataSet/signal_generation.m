function y=signal_generation(fb,fs,SNR,mod_type,time_sample)
% 发射按照4倍采样
nsamp_tx=fs/fb;
% 最后一个4是群时延
hcos_tx=rcosine(1,4,'fir/sqrt',0.25,4);
% 计算采样点数量
num_sapmle=floor(fs*time_sample);

num_bit=time_sample*fb*8;
% 定义各类调制器对象
mskMod= comm.MSKModulator('BitInput', true, ...
    'InitialPhaseOffset', pi/4,'SamplesPerSymbol',nsamp_tx);
gmskMod = comm.GMSKModulator('BitInput',true,'PulseLength',1, ...
    'SamplesPerSymbol',nsamp_tx,'BandwidthTimeProduct',0.3);
oqpskmod = comm.OQPSKModulator('SamplesPerSymbol',4,'RolloffFactor',0.25,'PulseShape','Root raised cosine');


% 生成不同调制类型对应的比特序列
if(mod_type=="oqpsk")
    x = randi([0 3],num_bit,1);%产生6000个比特
elseif(mod_type=="soqpsk")
    x = randi([0 3],num_bit,1);%产生6000个比特
elseif(mod_type=="8psk")
    x = randi([0 7],num_bit,1);%产生6000个比特
elseif(mod_type=="16qam")
    x = randi([0 15],num_bit,1);%产生6000个比特
else
    x = randi([0 1],num_bit,1);%产生6000个比特
end

% 根据调制类型进行信号调制与成型滤波
if(mod_type=="msk")
    modSignal= step(mskMod, x);
elseif(mod_type=="gmsk")
    modSignal= step(gmskMod, x);
elseif(mod_type=="oqpsk")
    modSignal_unsample= oqpskmod(x);
    modSignal=resample(modSignal_unsample,fs,fb*4);
elseif(mod_type=="bpsk")
    modSignal_unsample= pskmod(x,2);
    modSignal_upsample=upsample(modSignal_unsample,4);%4倍采样
    modSignal_filter=conv(modSignal_upsample,hcos_tx);%成型滤波
    modSignal=resample(modSignal_filter,fs,fb*4);
elseif(mod_type=="pi/4-bpsk")
    modSignal_unsample= pskmod(x,2,pi/4);
    modSignal_upsample=upsample(modSignal_unsample,4);%4倍采样
    modSignal_filter=conv(modSignal_upsample,hcos_tx);%成型滤波
    modSignal=resample(modSignal_filter,fs,fb*4);
elseif(mod_type=="8psk")
    modSignal_unsample= pskmod(x,8);
    modSignal_upsample=upsample(modSignal_unsample,4);%4倍采样
    modSignal_filter=conv(modSignal_upsample,hcos_tx);%成型滤波
    modSignal=resample(modSignal_filter,fs,fb*4);
elseif(mod_type=="16qam")
    modSignal_unsample= qammod(x,16);
    modSignal_upsample=upsample(modSignal_unsample,4);%4倍采样
    modSignal_filter=conv(modSignal_upsample,hcos_tx);%成型滤波
    modSignal=resample(modSignal_filter,fs,fb*4);
elseif(mod_type=="soqpsk")
    % 产生QPSK信号
    modSignal_qpsk= pskmod(x,4);
    mod_I=real(modSignal_qpsk);
    mod_I_upsample=upsample(mod_I,4);%4倍采样
    mod_Q=imag(modSignal_qpsk);
    mod_Q_upsample=upsample(mod_Q,4);%4倍采样
    mod_I_upsample=[mod_I_upsample;0;0];
    mod_Q_upsample=[0;0;mod_Q_upsample];
    modSignal_upsample=mod_I_upsample+sqrt(-1)*mod_Q_upsample;
    modSignal_filter=conv(modSignal_upsample,hcos_tx);%成型滤波
    modSignal=resample(modSignal_filter,fs,fb*4);
end

% 模拟随机起始采样
sample_start=randi([1 length(modSignal)-num_sapmle]);
SampleSignal=modSignal(sample_start:sample_start+num_sapmle-1);

% 计算信号功率与噪声功率
abs_SampleSignal=abs(SampleSignal);
pow_SampleSignal=sum(abs_SampleSignal.^2);%幅度的平方算功率
SNR_linear=10^(SNR/10);
N0=pow_SampleSignal/SNR_linear/length(SampleSignal);%每个采样点的噪声功率
sigma=sqrt(N0/2);
% 生成复高斯白噪声
noise=sigma*randn(length(SampleSignal),1)+sigma*sqrt(-1)*randn(length(SampleSignal),1);

% 输出最终带噪信号
y = SampleSignal + noise;
end