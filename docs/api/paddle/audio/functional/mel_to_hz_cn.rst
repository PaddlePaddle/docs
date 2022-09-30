.. _cn_api_audio_functional_mel_to_hz:

mel_to_hz
-------------------------------

.. py:function:: paddle.audio.functional.mel_to_hz(feq, htk=False)

转换Mels为Hz。

参数
::::::::::::

    - **mel** (Tensor, float) - 输入tensor。
    - **htk** (bool) - 是否使用htk缩放, 默认False。

返回
:::::::::

``paddle.Tensor或float``, hz为单位的频率。

代码示例
:::::::::

::

    import paddle

    val = 3.0

    htk_flag = True

    mel_paddle_tensor = paddle.audio.functional.mel_to_hz(
        paddle.to_tensor(val), htk_flag)
