.. _cn_api_audio_features_Spectrogram:

Spectrogram
-------------------------------

.. py:class:: paddle.audio.features.Spectrogram(n_fft=512, hop_length=512, win_length=None, window='hann', power=1.0, center=True, pad_mode='reflect', dtype='float32')

通过给定信号的短时傅里叶变换得到频谱。

参数
::::::::::::

    - **n_fft** (int) - 离散傅里叶变换中频率窗大小，默认 512。
    - **hop_length**  (int，可选) - 帧移，默认 512。
    - **win_length**  (int，可选) - 短时 FFT 的窗长，默认为 None。
    - **window**  (str) - 窗函数名，默认'hann'。
    - **power**  (float) - 幅度谱的指数。
    - **center**  (bool) - 对输入信号填充，如果 True，那么 t 以 t*hop_length 为中心，如果为 False，则 t 以 t*hop_length 开始。
    - **pad_mode**  (str) - 如果 center 是 True，选择填充的方式.默认值是'reflect'。
    - **dtype**  (str) - 输入和窗的数据类型，默认是'float32'。


返回
:::::::::

计算``Spectrogram``的可调用对象.

代码示例
:::::::::
COPY-FROM: paddle.audio.features.Spectrogram
