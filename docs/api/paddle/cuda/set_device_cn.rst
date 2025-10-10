.. _cn_api_paddle_cuda_set_device:

set_device
----------

.. py:function:: paddle.cuda.set_device(device)

设置当前设备。

参数
::::::::::::

    - **device** (DeviceLike) - 要设置的设备，可以是 "int" 用来表示设备 id，可以是形如 "gpu:0" 之类的设备描述字符串，也可以是 `paddle.CUDAPlace(0)` 之类的设备实例。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.set_device
