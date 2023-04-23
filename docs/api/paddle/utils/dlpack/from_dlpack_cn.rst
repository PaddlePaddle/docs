.. _cn_api_paddle_utils_dlpack_from_dlpack:

from_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.from_dlpack(dlpack)

将 DLPack 解码为 Tensor 对象。其中，DLPack 是一种开放的内存 Tensor 结构，可用于不同深度学习框架之间的 Tensor 共享。注意，一个 DLPack 只能被解码一次。

参数
:::::::::
  - **dlpack** (PyCapsule) - DLPack，即带有 dltensor 的 PyCapsule 对象。

返回
:::::::::
  - **out** (Tensor) - 从 DLPack 中解码得到的 Tensor。需要注意的是，对于带有`bool`数据类型的 dltensor 输入，我们最终解码得到的 Tensor 对应的数据类型为`uint8`。

代码示例
:::::::::
COPY-FROM: paddle.utils.dlpack.from_dlpack
