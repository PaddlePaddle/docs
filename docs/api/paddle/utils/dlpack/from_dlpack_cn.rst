.. _cn_api_paddle_utils_dlpack_from_dlpack:

from_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.from_dlpack(dlpack)

将DLPack解码为Tensor对象。其中，DLPack是一种开放的内存张量结构，可用于不同深度学习框架之间的张量共享。注意，一个DLPack只能被解码一次。

参数
:::::::::
  - **dlpack** (PyCapsule) - DLPack，即带有dltensor的PyCapsule对象。

返回
:::::::::
  - **out** (Tensor) - 从DLPack中解码得到的Tensor。需要注意的是，对于带有`bool`数据类型的dltensor输入，我们最终解码得到的Tensor对应的数据类型为`uint8`。

代码示例
:::::::::
.. code-block:: python

    import paddle 
    x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                          [0.1, 0.2, 0.6, 0.7]])
    dlpack = paddle.utils.dlpack.to_dlpack(x)
    x = paddle.utils.dlpack.from_dlpack(dlpack)
    print(x)
    # Tensor(shape=[2, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #         [[0.20000000, 0.30000001, 0.50000000, 0.89999998],
    #          [0.10000000, 0.20000000, 0.60000002, 0.69999999]]) 
