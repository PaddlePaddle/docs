.. _cn_api_tensor_sum:

sum
-------------------------------

.. py:function:: paddle.sum(input, dim=None, dtype=None, keep_dim=False, name=None)

:alias_main: paddle.sum
:alias: paddle.sum,paddle.tensor.sum,paddle.tensor.math.sum
:update_api: paddle.fluid.layers.reduce_sum



该OP是对指定维度上的Tensor元素进行求和运算，并输出相应的计算结果。

参数：
    - **input** （Variable）- 输入变量为多维Tensor或LoDTensor，支持数据类型为float32，float64，int32，int64。
    - **dim** （list | int ，可选）- 求和运算的维度。如果为None，则计算所有元素的和并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将变为 :math:`rank+dim[i]` ，默认值为None。
    - **dtype** （str ， 可选）- 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为None。
    - **keep_dim** （bool）- 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：  在指定dim上进行求和运算的Tensor，数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    # x是一个Tensor，元素如下:
    #    [[0.2, 0.3, 0.5, 0.9]
    #     [0.1, 0.2, 0.6, 0.7]]
    # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
    x = paddle.data(name='x', shape=[2, 4], dtype='float32')
    out1 = paddle.sum(x)
    out2 = paddle.sum(x, dim=0)
    out3 = paddle.sum(x, dim=-1)
    out4 = paddle.sum(x, dim=1, keep_dim=True)
    
    # y 是一个shape为[2, 2, 2]的Tensor元素如下:
    #      [[[1, 2], [3, 4]],
    #      [[5, 6], [7, 8]]]
    # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
    y = paddle.data(name='y', shape=[2, 2, 2], dtype='float32')
    out5 = paddle.sum(y, dim=[1, 2])
    out6 = paddle.sum(y, dim=[0, 1])

