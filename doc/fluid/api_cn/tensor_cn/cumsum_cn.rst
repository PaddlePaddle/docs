.. _cn_api_tensor_cn_cumsum:

cumsum
-------------------------------

.. py:function:: paddle.cumsum(x, axis=None, dtype=None, name=None)



沿给定 ``axis`` 计算张量 ``x`` 的累加和。结果的第一个元素和输入的第一个元素相同。

参数：
    - **x** (Tensor) - 累加的输入，需要进行累加操作的Tensor.
    - **axis** (int，可选) - 指明需要累加的维度。-1代表最后一维。默认：None，将输入展开为一维变量再进行累加计算。
    - **dtype** (str，可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64. 如果指定了，那么在执行操作之前，输入张量将被转换为dtype. 这对于防止数据类型溢出非常有用。默认为：None.
    - **name** （str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：累加的结果，即累加器的输出。

返回类型：Tensor

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.imperative import to_variable
    import numpy as np

    paddle.enable_imperative()
    data_np = np.arange(12).reshape(3, 4)
    data = to_variable(data_np)

    y = paddle.cumsum(data)
    print(y.numpy())
    # [ 0  1  3  6 10 15 21 28 36 45 55 66]

    y = paddle.cumsum(data, axis=0)
    print(y.numpy())
    # [[ 0  1  2  3]
    #  [ 4  6  8 10]
    #  [12 15 18 21]]
    
    y = paddle.cumsum(data, axis=-1)
    print(y.numpy())
    # [[ 0  1  3  6]
    #  [ 4  9 15 22]
    #  [ 8 17 27 38]]

    y = paddle.cumsum(data, dtype='float64')
    print(y.dtype)
    # VarType.FP64


