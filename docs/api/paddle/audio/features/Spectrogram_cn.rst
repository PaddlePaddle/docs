.. _cn_api_audio_features_Spectrogram:

Spectrogram
-------------------------------

.. py:class:: paddle.audio.features.Spectrogram(n_fft=512, hop_length=512, win_length=None, window='hann', power=1.0, center=True, pad_mode='reflect', dtype='float32')

通过给定信号的短时傅里叶变换得到频谱。

参数
::::::::::::

    - **n_fft** (int) - 离散傅里叶变换中频率窗大小，默认512。
    - **hop_length**  (int，可选) - 帧移，默认512。
    - **win_length**  (int，可选) - 短时FFT的窗长，默认为None。
    - **window**  (str) - 窗函数名，默认'hann'。
    - **power**  (float) - 幅度谱的指数。
    - **center**  (bool) - 对输入信号填充，如果True，那么t以t*hop_length为中心，如果为False，则t以t*hop_length开始。
    - **pad_mode**  (str) - 如果center是True，选择填充的方式.默认值是'reflect'。
    - **dtype**  (str) - 输入和窗的数据类型，默认是'float32'。


返回
:::::::::

计算``Spectrogram``的可调用对象.

代码示例
:::::::::
COPY-FROM: paddle.audio.features.Spectrogram
