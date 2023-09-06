.. _cn_api_paddle_utils_dlpack_to_dlpack:

to_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.to_dlpack(x)

将 Tensor 对象转化为 DLPack。其中，DLPack 是一种开放的内存 Tensor 结构，可用于不同深度学习框架之间的 Tensor 共享。

参数
:::::::::
  - **x** (Tensor) - Paddle Tensor，并且其数据类型为支持 bool，float16，float32，float64，int8，int16，int32，int64，uint8，complex64，complex128。

返回
:::::::::
  - **dlpack** (PyCapsule) - DLPack，即带有 dltensor 的 PyCapsule 对象。

代码示例
:::::::::
COPY-FROM: paddle.utils.dlpack.to_dlpack
