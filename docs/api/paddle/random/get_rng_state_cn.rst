.. _cn_api_paddle_random_get_rng_state:

get_rng_state
-------------------------------

.. py:function:: paddle.random.get_rng_state(device=None)

获取默认随机数生成器的随机状态。

参数
:::::::::

    - **device** (_CPUPlaceLike|None，可选) - 要获取随机状态的设备。默认值为 None。

返回
:::::::::
Tensor：随机状态 Tensor。
