.. _cn_api_amp_is_float16_supported:

is_float16_supported
-------------------------------

.. py:function:: paddle.amp.is_float16_supported(place=None)


在自动混合精度策略（AMP）场景下判断当前设备是否支持 float16。

参数
::::::::::::

    - **place** (fluid.CPUPlace|fluid.CUDAPlace|None, optional) - 需要查询的设备。默认为 None。


代码示例
:::::::::
COPY-FROM: paddle.amp.is_float16_supported
