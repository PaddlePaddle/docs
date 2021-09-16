.. _cn_api_paddle_utils_dlpack_from_dlpack:

from_dlpack
-------------------------------

.. py:function:: paddle.utils.dlpack.from_dlpack(dlpack)

将DLPack解码为Tensor对象。注意一个DLPack只能被解码一次。

参数：
  - **dlpack** (PyCapsule) - DLPack，即带有dltensor的PyCapsule对象。

返回：
  - **out** (Tensor) - 从DLPack中解码得到的Tensor。

**代码示例**

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
