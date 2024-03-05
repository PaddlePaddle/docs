.. _cn_api_paddle_audio_functional_mel_frequencies:

mel_frequencies
-------------------------------

.. py:function:: paddle.audio.functional.mel_frequencies(n_mels=64, f_min=0.0, f_max=11025, htk=False, dtype='float32')

计算 Mels 频率。

参数
::::::::::::

    - **n_mels** (int，可选) - 输入 tensor，默认 64。
    - **f_min** (float，可选) - 最小频率(hz)，默认 0.0。
    - **f_max** (float，可选) - 最大频率(hz)，默认 11025.0。
    - **htk** (bool，可选) - 是否使用 htk 缩放，默认 False。
    - **dtype** (str，可选) - 默认'float32'。

返回
:::::::::

``paddle.Tensor``，Tensor 形状 (n_mels,)。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.mel_frequencies
