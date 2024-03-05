.. _cn_api_paddle_audio_functional_fft_frequencies:

fft_frequencies
-------------------------------

.. py:function:: paddle.audio.functional.fft_frequencies(sr, n_fft, dtype='float32')

计算 fft 频率。

参数
::::::::::::

    - **sr** (int) - 采样率。
    - **n_fft** (int) - fft bins 的数目。
    - **dtype** (str，可选) - 默认'float32'。

返回
:::::::::

``paddle.Tensor``，Tensor 形状 (n_fft//2 + 1,)。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.fft_frequencies
