.. _cn_api_paddle_device_get_rng_state:

get_rng_state
-------------------------------

.. py:function:: paddle.device.get_rng_state(device=None)

返回指定设备的随机数生成器状态，以 GeneratorState 形式表示。

参数
:::::::::
- **device** (DeviceLike, 可选) - 要获取 RNG 状态的设备:

  - 如果不指定，则使用当前默认设备(由 paddle.framework._current_expected_place_() 返回)
  - 可以是设备对象、整数设备 ID 或设备字符串

返回
:::::::::
core.GeneratorState: 指定设备的当前 RNG 状态，以 GeneratorState 形式表示。

代码示例
::::::::::::
COPY-FROM: paddle.device.get_rng_state
