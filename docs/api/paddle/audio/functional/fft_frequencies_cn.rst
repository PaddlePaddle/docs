.. _cn_api_audio_functional_fft_frequencies:

fft_frequencies
-------------------------------

.. py:function:: paddle.audio.functional.fft_frequencies(sr, n_fft, dtype='float32')

计算fft频率。

参数
::::::::::::

    - **sr** (int) - 采样率。
    - **n_fft** (int) - fft bins的数目。
    - **dtype** (str) - 默认'float32'。

返回
:::::::::

``paddle.Tensor``,Tensor shape (n_fft//2 + 1,)。

代码示例
:::::::::

::
    import paddle

    sr = 16000

    n_fft = 128

    fft_freq = paddle.audio.functional.fft_frequencies(sr, n_fft)
