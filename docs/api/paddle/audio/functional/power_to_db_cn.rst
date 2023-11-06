.. _cn_api_paddle_audio_functional_power_to_db:

power_to_db
-------------------------------

.. py:function:: paddle.audio.functional.power_to_db(spect, ref_value=1.0, amin=1e-10, top_db=80.0)

转换能量谱为分贝单位。

参数
::::::::::::

    - **spect** (Tensor) - stft 能量谱，输入 tensor。
    - **ref_value** (float，可选) - 参照值，振幅相对于 ref 进行缩放，默认 1.0。
    - **amin** (float，可选) - 最小阈值，默认 1e-10。
    - **top_db** (float，可选) - 阈值，默认 80.0。

返回
:::::::::

``paddle.Tensor 或 float``，db 单位的能量谱。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.power_to_db
