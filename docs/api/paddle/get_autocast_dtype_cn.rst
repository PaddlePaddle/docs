.. _cn_api_paddle_get_autocast_dtype:

get_autocast_dtype
-------------------------------

.. py:function:: paddle.get_autocast_dtype(device_type=None)




获取当前上下文中自动混合精度的默认数据类型。


参数
::::::::::::

    - **device_type** (str, 可选) - 设备类型，默认为 None。注意：在 PaddlePaddle 中所有设备共享同一套自动混合精度配置，实际返回值与设备类型无关。

返回
::::::::::::
字符串，当前上下文中自动混合精度的默认数据类型。


代码示例
::::::::::::

COPY-FROM: paddle.get_autocast_gpu_dtype
