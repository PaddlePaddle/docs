.. _cn_api_paddle_from_dlpack:

from_dlpack
-------------------------------

.. py:function:: paddle.from_dlpack(dlpack, *, device=None, copy=None)

将 DLPack 格式的 Tensor 解码为 Paddle Tensor，DLPACK 是一种通用稳定的内存数据结构，可用于不同深度学习框架之间的 Tensor 共享。

.. note::
    一个 dlpack 只能被  ``from_dlpack``  解码一次。

参数
:::::::::
  - **dlpack** (SupportDLPack | PyCapsule) - 一个实现了  ``__dlpack__``  与  ``__dlpack_device__``  方法的对象，或者是一个带有 dltensor 的  ``PyCapsule``  对象。
  - **device** (PlaceLike, 可选) - 返回的 Tensor 所在的设备。如果不指定，返回的 Tensor 将与输入的 `dlpack` 位于同一设备上。
  - **copy** (bool, 可选) - 是否复制输入的内容。如果为 True，则输出的 Tensor 总是被复制。如果为 False，则输出的 Tensor 永远不会被复制，如果需要复制则会引发  ``BufferError``  异常。如果为 None，则在可能的情况下重用现有的内存缓冲区，否则进行复制。默认值为 None。

返回
:::::::::
  - **out** (Tensor) - 从 `dlpack` 中解码得到的 Paddle Tensor，支持的数据类型为： bool，float16，float32，float64，uint8，int8，int16，int32，int64，complex64，complex128，支持的设备类型为：  ``CPU`` ，  ``CUDAPlace`` ，  ``CUDAPinnedPlace`` 。

代码示例 1
:::::::::

COPY-FROM: paddle.utils.dlpack.from_dlpack:code-paddle-from-paddle

代码示例 2
:::::::::

COPY-FROM: paddle.utils.dlpack.from_dlpack:code-paddle-from-numpy
