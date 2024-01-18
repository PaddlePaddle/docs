.. _cn_api_paddle_get_rng_state:

get_rng_state
-------------------------------
.. py:function:: paddle.get_rng_state(device=None)

获取指定设备的随机数生成器的所有随机状态。

参数
::::::::::::
    - **device** (str) - 此参数决定了具体的运行设备。可以是 ``cpu``、``gpu``、``xpu``。默认值为 None。如果为 None，则返回当前设备（由 ``set_device`` 指定）的生成器。

返回
::::::::::::
    - GeneratorState：对象。

代码示例
::::::::::::

COPY-FROM: paddle.get_rng_state
