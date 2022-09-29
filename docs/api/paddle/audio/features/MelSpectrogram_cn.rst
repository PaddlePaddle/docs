.. _cn_api_audio_features_MelSpectrogram:

MelSpectrogram
-------------------------------

.. py:class:: paddle.audio.features.MelSpectrogram(sr=22050, n_fft=2048, hop_length=512, win_length=None, window='hann', power=2.0, center=True, pad_mode='reflect', n_mels=64, f_min=50.0, f_max=None, htk=False, norm='slaney', dtype='float32')

求得给定信号的Mel谱。

参数
::::::::::::

    - **sr** (int，可选) - 采样率，默认22050。
    - **n_fft** (int) - 离散傅里叶变换中频率窗大小，默认512。
    - **hop_length**  (int，可选) - 帧移，默认512。
    - **win_length**  (int，可选) - 短时FFT的窗长，默认为None。
    - **window**  (str) - 窗函数名，默认'hann'。
    - **power**  (float) - 幅度谱的指数。
    - **center**  (bool) - 对输入信号填充，如果True，那么t以t*hop_length为中心，如果为False，则t以t*hop_length开始。
    - **pad_mode**  (str) - 如果center是True，选择填充的方式.默认值是'reflect'。
    - **n_mels** (int) - mel bins的数目。
    - **f_min** (float，可选) - 最小频率(hz)，默认 50.0。
    - **f_max** (float，可选) - 最大频率(hz)，默认为None。
    - **htk** (bool，可选) - 在计算fbank矩阵时是否用在HTK公式缩放。
    - **norm** (Union[str，float]，可选) -计算fbank矩阵时正则化的种类，默认是'slaney'，你也可以norm=0.5，使用p-norm正则化。
    - **dtype**  (str) - 输入和窗的数据类型，默认是'float32'。


返回
:::::::::

计算``MelSpectrogram``的可调用对象。

代码示例
:::::::::

::
    import paddle
    from paddle.audio.features import MelSpectrogram
    
    sample_rate = 16000 
    wav_duration = 0.5
    num_channels = 1
    num_frames = sample_rate * wav_duration
    wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
    waveform = wav_data.tile([num_channels, 1])

    feature_extractor = MelSpectrogram(sr=sample_rate, n_fft=512, window = 'hann')
    feats = feature_extractor(waveform)
