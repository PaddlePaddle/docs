.. _cn_api_paddle_tensor_creation_assign:

assign
-------------------------------

.. py:function:: paddle.assign(x,output=None)




该OP将输入Tensor或numpy数组拷贝至输出Tensor。

参数：
    - **x** (Tensor|np.ndarray|list|tuple|scalar) - 输入Tensor，或numpy数组，或由基本数据组成的list/tuple，或基本数据，支持数据类型为float32, float64, int32, int64和bool。注意：由于当前框架的protobuf传输数据限制，float64数据会被转化为float32数据。
    - **output** (Tensor，可选) - 输出Tensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。

返回：输出Tensor，形状、数据类型、数据值和 ``x`` 一致。


**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    array = np.array([[1, 1],
                      [3, 4],
                      [1, 3]]).astype(np.int64)
    result1 = paddle.zeros(shape=[3, 3], dtype='float32')
    paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
    result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
