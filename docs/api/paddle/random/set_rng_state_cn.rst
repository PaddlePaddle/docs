.. _cn_api_paddle_random_set_rng_state:

set_rng_state
-------------------------------

.. py:function:: paddle.random.set_rng_state(new_state, device=None)

设置指定设备的随机数生成器状态。

参数
:::::::::
    - **new_state** (core.GeneratorState) - 要设置的随机数生成器状态，应由 ``get_rng_state()`` 获得。
    - **device** (DeviceLike，可选) - 要设置随机状态的设备。未指定时使用当前默认设备；可以是设备对象、整数设备 ID 或设备字符串。默认值为 None。


返回
:::::::::
无
