.. _cn_api_audio_functional_hz_to_mel:

hz_to_mel
-------------------------------

.. py:function:: paddle.audio.functional.hz_to_mel(feq, htk=False)

转换Hz为Mels。

参数
::::::::::::

    - **freq** (Tensor, float) - 输入tensor。
    - **htk** (bool) - 是否使用htk缩放, 默认False。

返回
:::::::::

``paddle.Tensor或float``, mels值。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.hz_to_mel
