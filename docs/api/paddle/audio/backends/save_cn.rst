.. _cn_api_audio_backends_save:

save
-------------------------------

.. py:function:: paddle.audio.backends.save(filepath: str, src: paddle.Tensor, sample_rate: int, channels_first: bool = True, encoding: Optional[str] = None, bits_per_sample: Optional[int] = 16)

获取音频的相关信息，如采用率，通道数等。

参数
::::::::::::

    - **filepath** (str 或者 Path) - 保存音频路径。
    - **src** (paddle.Tensor) - 音频数据。
    - **sample_rate** (int) - 采样率。
    - **channels_first** (bool) - 如果是True，那么src的Tensor形状是[channel，time]，如果是False，则是[time, channel]。
    - **encoding** (Optional[str]) - 默认是None，编码信息。
    - **bits_per_sample** (Optional[int]) - 默认是16。编码位长。
返回
:::::::::

代码示例
:::::::::

COPY-FROM: paddle.audio.backends.save