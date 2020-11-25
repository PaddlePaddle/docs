.. _cn_api_tensor_cn_prod:

prod
-------------------------------

.. py:function:: paddle.prod(x, axis=None, keepdim=False, dtype=None, name=None)



对指定维度上的Tensor元素进行求乘积运算，并输出相应的计算结果。

参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、int32、int64。
    - **axis** (int|list|tuple，可选) - 求乘积运算的维度。如果是None，则计算所有元素的乘积并返回包含单个元素的Tensor，否则该参数必须在 :math:`[-x.ndim, x.ndim)` 范围内。如果 :math:`axis[i] < 0` ，则维度将变为 :math:`x.ndim + axis[i]` ，默认为None。
    - **keepdim** (bool，可选) - 是否在输出 `Tensor` 中保留减小的维度。如 `keepdim` 为True，否则结果张量的维度将比输入张量小，默认值为False。
    - **dtype** (str，可选) - 输出Tensor的数据类型，支持int32、int64、float32、float64。如果指定了该参数，那么在执行操作之前，输入Tensor将被转换为dtype类型. 这对于防止数据类型溢出非常有用。若参数为空，则输出变量的数据类型和输入变量相同，默认为：None。
    - **name** （str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：指定axis上累乘的结果的Tensor。
    
    
**代码示例**：
    
.. code-block:: python 
    
    import paddle
    import numpy as np

    
    # the axis is a int element
    data_x = np.array([[0.2, 0.3, 0.5, 0.9],
                 [0.1, 0.2, 0.6, 0.7]]).astype(np.float32)
    x = paddle.to_tensor(data_x)
    out1 = paddle.prod(x)
    # [0.0002268]
    
    out2 = paddle.prod(x, -1)
    # [0.027  0.0084]

    out3 = paddle.prod(x, 0)
    # [0.02 0.06 0.3  0.63]

    out4 = paddle.prod(x, 0, keepdim=True)
    # [[0.02 0.06 0.3  0.63]]

    out5 = paddle.prod(x, 0, dtype='int64')
    # [0 0 0 0]

    # the axis is list
    data_y = np.array([[[1.0, 2.0], [3.0, 4.0]],
                       [[5.0, 6.0], [7.0, 8.0]]])
    y = paddle.to_tensor(data_y)
    out6 = paddle.prod(y, [0, 1])
    # [105. 384.]

    out7 = paddle.prod(y, (1, 2))
    # [  24. 1680.]
