.. _cn_api_audio_save:

save
-------------------------------

.. py:function:: paddle.audio.save(filepath: str, src: paddle.Tensor, sample_rate: int, channels_first: bool = True, encoding: Optional[str] = None, bits_per_sample: Optional[int] = 16)

保存音频数据。

参数
::::::::::::

    - **filepath** (str 或者 Path) - 保存音频路径。
    - **src** (paddle.Tensor) - 音频数据。
    - **sample_rate** (int) - 采样率。
    - **channels_first** (bool，可选) - 如果是 True，那么 src 的 Tensor 形状是[channel，time]，如果是 False，则是[time，channel]。
    - **encoding** (Optional[str]，可选) - 默认是 None，编码信息。
    - **bits_per_sample** (Optional[int]，可选) - 默认是 16，编码位长。
返回
:::::::::
无

代码示例
:::::::::

COPY-FROM: paddle.audio.save
