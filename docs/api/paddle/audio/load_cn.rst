.. _cn_api_paddle_audio_loa:

load
-------------------------------

.. py:function:: paddle.audio.load(filepath: Union[str, Path], frame_offset: int = 0, num_frames: int = -1, normalize: bool = True, channels_first: bool = True)

获取音频数据。

参数
::::::::::::

    - **filepath** (str 或者 Path) - 输入音频路径。
    - **frame_offset** (int，可选) - 默认是 0，开始读取音频起始帧。
    - **num_frames** (int，可选) - 默认是-1，读取音频帧数，-1 表示读取全部帧。
    - **normalize** (bool，可选) - 默认是 True。如果是 True，返回是音频值被规整到[-1.0，1.0]，如果是 False，那么就返回原始值。
    - **channels_first** (bool，可选) - 默认是 True。如果是 True，那么返回的形状是[channel，time]，如果是 False，则是[time，channel]。
返回
:::::::::

``Tuple[paddle.Tensor, int]``，音频数据值，采样率。

代码示例
:::::::::

COPY-FROM: paddle.audio.load
