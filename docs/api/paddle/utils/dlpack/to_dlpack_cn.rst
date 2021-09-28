.. _cn_api_paddle_utils_dlpack_to_dlpack:

to_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.to_dlpack(x)

将Tensor对象转化为DLPack。其中，DLPack是一种开放的内存张量结构，可用于不同深度学习框架之间的张量共享。

参数：
  - **x** (Tensor) - Paddle Tensor，并且其数据类型为支持bool，float16，float32，float64，int8，int16，int32，int64，uint8，complex64，complex128。

返回：
  - **dlpack** (PyCapsule) - DLPack，即带有dltensor的PyCapsule对象。


**代码示例**

.. code-block:: python

    import paddle
    x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                          [0.1, 0.2, 0.6, 0.7]])
    dlpack = paddle.utils.dlpack.to_dlpack(x)
    print(dlpack)
    # <capsule object "dltensor" at 0x7f6103c681b0>
