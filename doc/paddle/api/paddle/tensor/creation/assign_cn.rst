.. _cn_api_fluid_layers_assign:

assign
-------------------------------

.. py:function:: paddle.assign(input,output=None)




该OP将输入Tensor或numpy数组拷贝至输出Tensor。

参数：
    - **input** (Tensor|np.ndarray) - 输入Tensor或numpy数组，支持数据类型为float32, float64, int32, int64和bool。
    - **output** (Tensor，可选) - 输出Tensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。

返回：输出Tensor，形状、数据类型、数据值和 ``input`` 一致。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    array = np.array([[1, 1],
                      [3, 4],
                      [1, 3]]).astype(np.int64)
    result1 = paddle.zeros(shape=[3, 3], dtype='float32')
    paddle.nn.functional.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
    result2 = paddle.nn.functional.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    result3 = paddle.nn.functional.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
