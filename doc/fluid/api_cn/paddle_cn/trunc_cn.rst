.. _cn_api_tensor_trunc:

trunc
-------------------------------

.. py:function:: paddle.trunc(input, name=None)


将输入矩阵数据的小数部分置0，返回置0后的矩阵，如果输入矩阵的数据类型为整数，则不做处理。


参数：
    - **input** (Tensor) : 输入变量，类型为 Tensor, 支持int、float、double数据类型。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
    - Tensor (Tensor)，矩阵截断后的结果。


**代码示例**：

.. code-block:: python

    import paddle

    input = paddle.rand([2,2],'float32')
    print(input)
    # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #         [[0.02331470, 0.42374918],
    #         [0.79647720, 0.74970269]])
            
    output = paddle.trunc(input)
    print(output)
    # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #         [[0., 0.],
    #         [0., 0.]])