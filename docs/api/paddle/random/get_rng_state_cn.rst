.. _cn_api_paddle_random_get_rng_state:

get_rng_state
-------------------------------

.. py:function:: paddle.random.get_rng_state(device=None)

返回默认 CPU 随机数生成器的状态。该接口是 ``paddle.device.cpu.get_rng_state`` 的别名。

参数
:::::::::

    - **device** (CPUPlace|str|int|None，可选) - 兼容设备参数；当前实现读取默认 CPU 生成器的状态。默认值：None。

返回
:::::::::
core.GeneratorState：随机数生成器状态，可传给 ``paddle.random.set_rng_state`` 恢复随机状态。
