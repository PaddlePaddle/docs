.. _cn_api_audio_functional_mel_to_hz:

mel_to_hz
-------------------------------

.. py:function:: paddle.audio.functional.mel_to_hz(feq, htk=False)

转换 Mels 为 Hz。

参数
::::::::::::

    - **mel** (Tensor, float) - 输入 tensor。
    - **htk** (bool) - 是否使用 htk 缩放, 默认 False。

返回
:::::::::

``paddle.Tensor 或 float``, hz 为单位的频率。

代码示例
:::::::::

::

    import paddle

    val = 3.0

    htk_flag = True

    mel_paddle_tensor = paddle.audio.functional.mel_to_hz(
        paddle.to_tensor(val), htk_flag)
