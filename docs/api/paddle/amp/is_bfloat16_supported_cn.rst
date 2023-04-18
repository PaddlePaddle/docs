.. _cn_api_amp_is_bfloat16_supported:

is_bfloat16_supported
-------------------------------

.. py:function:: paddle.amp.is_bfloat16_supported(device=None)


在自动混合精度策略（AMP）场景下判断设备是否支持 bfloat16。

参数
::::::::::::

    - **device** (str|None, optional) - 指定要查询的设备，它可以是 cpu、 gpu、 xpu、gpu:x、xpu:x。其中，x 是 GPU、 XPU 的编号。如果 ``device`` 为 None， 则查询当前设备（与飞桨安装版本保持一致），默认为 None。


代码示例
:::::::::
COPY-FROM: paddle.amp.is_bfloat16_supported
