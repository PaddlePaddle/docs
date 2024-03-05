.. _cn_api_paddle_audio_features_MFCC:

MFCC
-------------------------------

.. py:class:: paddle.audio.features.MFCC(sr=22050, n_mfcc=40, n_fft=2048, hop_length=512, win_length=None, window='hann', power=2.0, center=True, pad_mode='reflect', n_mels=64, f_min=50.0, f_max=None, htk=False, norm='slaney', ref_value=1.0, amin=1e-10, top_db=None, dtype='float32')

计算给定信号的 MFCC。

参数
::::::::::::

    - **sr** (int，可选) - 采样率，默认 22050。
    - **n_mfcc** (int，可选) - mfcc 的维度，默认 40。
    - **n_fft** (int，可选) - 离散傅里叶变换中频率窗大小，默认 512。
    - **hop_length**  (int，可选) - 帧移，默认 512。
    - **win_length**  (int，可选) - 短时 FFT 的窗长，默认为 None。
    - **window**  (str，可选) - 窗函数名，默认'hann'。
    - **power**  (float，可选) - 幅度谱的指数，默认是 2.0。
    - **center**  (bool，可选) - 对输入信号填充，如果 True，那么 t 以 t*hop_length 为中心，如果为 False，则 t 以 t*hop_length 开始，默认是 True。
    - **pad_mode**  (str，可选) - 如果 center 是 True，选择填充的方式，默认值是'reflect'。
    - **n_mels** (int，可选) - mel bins 的数目，默认是 64。
    - **f_min** (float，可选) - 最小频率(hz)，默认 50.0。
    - **f_max** (float，可选) - 最大频率(hz)，默认为 None。
    - **htk** (bool，可选) - 在计算 fbank 矩阵时是否用在 HTK 公式缩放，默认是 False。
    - **norm** (Union[str, float]，可选) - 计算 fbank 矩阵时正则化的种类，默认是'slaney'，也可以 norm=0.5，使用 p-norm 正则化。
    - **ref_value** (float，可选) - 参照值， 如果小于 1.0，信号的 db 会被提升， 相反 db 会下降， 默认值为 1.0。
    - **amin** (float，可选) - 输入的幅值的最小值，默认是 1e-10。
    - **top_db** (float，可选) - log-mel 谱的最大值(db)，默认是 None。
    - **dtype**  (str，可选) - 输入和窗的数据类型，默认是'float32'。

返回
:::::::::

计算``MFCC``的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.audio.features.layers.MFCC
