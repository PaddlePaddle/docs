.. _cn_api_audio_functional_compute_fbank_matrix:

compute_fbank_matrix
-------------------------------

.. py:function:: paddle.audio.functional.compute_fbank_matrix(sr, n_fft, n_mels=64, f_min=0.0, f_max=None, htk=False, nrom='slaney', dtype='float32')

计算 mel 变换矩阵。

参数
::::::::::::

    - **sr** (int) - 采样率。
    - **n_fft** (int) - fft bins 的数目。
    - **n_mels** (float) - mels bins 的数目。
    - **f_min** (float) - 最小频率(hz)。
    - **f_max** (Optional[float]) -最大频率(hz)。
    - **htk** (bool) -是否使用 htk 缩放。
    - **norm** (Union[str，float]) -norm 的类型，默认是'slaney'。
    - **dtype** (str) - 返回矩阵的数据类型，默认'float32'。

返回
:::::::::

``paddle.Tensor``,Tensor shape (n_mels, n_fft//2 + 1)。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.compute_fbank_matrix
