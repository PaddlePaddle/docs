.. _cn_api_paddle_set_rng_state:

set_rng_state
-------------------------------

.. py:function:: paddle.set_rng_state(state_list, device=None)

为所有设备生成器设置生成器状态。

参数
::::::::::::
    - **state_list** (list|tuple) - 要设置回设备生成器的设备状态。state_list 是从 ``get_rng_state()`` 获取的。
    - **device** (str) - 此参数决定了具体的运行设备。可以是 ``cpu``、``gpu``、``xpu``。默认值为 None。如果为 None，则返回当前设备（由 ``set_device`` 指定）的生成器。

返回
::::::::::::
    - 无。


代码示例
::::::::::::

COPY-FROM: paddle.set_rng_state
