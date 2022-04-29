.. _cn_api_paddle_utils_run_check:

run_check
-------------------------------

.. py:function:: paddle.utils.run_check(custom_device_type=None)

检查用户机器上，PaddlePaddle或者自定义硬件插件（custom_device_type为None时）是否正确地安装了，以及是否能够成功运行。

参数
:::::::::
    - **custom_device_type** (str，可选） - 指定要检查的自定义硬件类型。如果设置，则检查插件，如果未位置，则检查PaddlePaddle。所有自定义硬件插件类型通过 paddle.device.get_all_custom_device_type() 获取。

代码示例
::::::::::
COPY-FROM: paddle.utils.run_check
