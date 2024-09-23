.. _cn_api_paddle_utils_dlpack_to_dlpack:

to_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.to_dlpack(x)

将 Tensor 对象转化为一个带有 dltensor 的 ``PyCapsule`` 对象，该对象是一种称为 DLPack 的通用稳定内存数据结构，可用于不同深度学习框架之间的 Tensor 共享。

参数
:::::::::
  - **x** (Tensor) - Paddle Tensor，支持的数据类型为： bool，float16，float32，float64，uint8，int8，int16，int32，int64，uint8，complex64，complex128。

返回
:::::::::
  - **dlpack** (PyCapsule) - 一个带有 dltensor 的 ``PyCapsule`` 对象。

代码示例 1
:::::::::

COPY-FROM: paddle.utils.dlpack.to_dlpack:paddle-to-paddle

代码示例 2
:::::::::

COPY-FROM: paddle.utils.dlpack.to_dlpack:paddle-to-torch
