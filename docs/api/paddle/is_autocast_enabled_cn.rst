.. _cn_api_paddle_is_autocast_enabled:

is_autocast_enabled
-------------------------------

.. py:function:: paddle.is_autocast_enabled(device_type=None)




获取当前上下文中是否启用了自动混合精度。


参数
::::::::::::

    - **device_type** (str, 可选) - 设备类型，默认为 None。注意：在 PaddlePaddle 中所有设备共享同一套自动混合精度配置，实际返回值与设备类型无关。

返回
::::::::::::
布尔值，如果当前上下文中启用了自动混合精度，则返回  ``True`` ，否则返回  ``False`` 。


代码示例
::::::::::::

COPY-FROM: paddle.is_autocast_enabled
