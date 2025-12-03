.. _cn_api_paddle_cuda_set_rng_state:

set_rng_state
-------------------------------

.. py:function:: paddle.cuda.set_rng_state(new_state, device=None)

设置指定设备的随机数生成器状态。

参数
:::::::::
- **new_state** (core.GeneratorState) - 要设置的 RNG 状态对象，通常从 ``get_rng_state()`` 获取
- **device** (DeviceLike, 可选) - 要设置 RNG 状态的设备:

  - 如果不指定，则使用当前默认设备(由 ``paddle.framework._current_expected_place_()`` 返回)
  - 可以是设备对象、整数设备 ID 或设备字符串

返回
:::::::::
None

代码示例
:::::::::
COPY-FROM: paddle.cuda.set_rng_state
