.. _cn_api_audio_functional_power_to_db:

power_to_db
-------------------------------

.. py:function:: paddle.audio.functional.power_to_db(spect, ref_value=1.0, amin=1e-10, top_db=80.0)

转换能量谱为分贝单位。

参数
::::::::::::

    - **spect** (Tensor) - stft能量谱, 输入tensor。
    - **ref_value** (float) - 参照值, 振幅相对于ref进行缩放, 默认 1.0。
    - **amin** (float) - 最小阈值, 默认 1e-10。
    - **top_db** (float，可选) - 阈值, 默认 80.0。

返回
:::::::::

``paddle.Tensor或float``, db单位的能量谱。

代码示例
:::::::::

::

    import paddle

    val = 3.0

    decibel_paddle = paddle.audio.functional.power_to_db(
        paddle.to_tensor(val))
