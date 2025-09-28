.. _cn_api_paddle_audio_functional_hz_to_mel:

hz_to_mel
-------------------------------

.. py:function:: paddle.audio.functional.hz_to_mel(feq, htk=False)

转换 Hz 为 Mels。

参数
::::::::::::

    - **freq** (Tensor, float) - 输入 tensor。
    - **htk** (bool，可选) - 是否使用 htk 缩放，默认 False。

返回
:::::::::

``paddle.Tensor 或 float``，mels 值。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.hz_to_mel
