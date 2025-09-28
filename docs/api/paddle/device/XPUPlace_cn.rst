.. _cn_api_paddle_device_XPUPlace:

XPUPlace
-------------------------------

.. py:class:: paddle.device.XPUPlace





``XPUPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 的 Baidu Kunlun XPU 设备。
每个 ``XPUPlace`` 有一个 ``dev_id`` （设备 id）来表明当前的 ``XPUPlace`` 所代表的显卡编号，编号从 0 开始。
``dev_id`` 不同的 ``XPUPlace`` 所对应的内存不可相互访问。

参数
::::::::::::

  - **id** (int，可选) - XPU 的设备 ID。如果为 ``None``，则默认会使用 id 为 0 的设备。默认值为 ``None``。

代码示例
::::::::::::

COPY-FROM: paddle.device.XPUPlace
