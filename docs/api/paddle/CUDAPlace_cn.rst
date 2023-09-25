.. _cn_api_paddle_CUDAPlace:

CUDAPlace
-------------------------------

.. py:class:: paddle.CUDAPlace





``CUDAPlace`` 是一个设备描述符，表示一个分配或将要分配 ``Tensor`` 的 GPU 设备。
每个 ``CUDAPlace`` 有一个 ``dev_id`` （设备 id）来表明当前的 ``CUDAPlace`` 所代表的显卡编号，编号从 0 开始。
``dev_id`` 不同的 ``CUDAPlace`` 所对应的内存不可相互访问。
这里编号指的是可见显卡的逻辑编号，而不是显卡实际的编号。
可以通过 ``CUDA_VISIBLE_DEVICES`` 环境变量限制程序能够使用的 GPU 设备，程序启动时会遍历当前的可见设备，并从 0 开始为这些设备编号。
如果没有设置 ``CUDA_VISIBLE_DEVICES``，则默认所有的设备都是可见的，此时逻辑编号与实际编号是相同的。

参数
::::::::::::

  - **id** (int) - GPU 的设备 ID。

代码示例
::::::::::::

COPY-FROM: paddle.CUDAPlace
