.. _cn_api_paddle_tensor_max:

max
-------------------------------

.. py:function:: paddle.tensor.max(x, axis=None, keepdim=False, name=None)

:alias_main: paddle.max
:alias: paddle.max,paddle.tensor.max,paddle.tensor.math.max

该OP是对指定维度上的Tensor元素求最大值运算，并输出相应的计算结果。

参数
:::::::::
   - **x** （Tensor）- Tensor，支持数据类型为float32，float64，int32，int64。
   - **axis** （list | int ，可选）- 求最大值运算的维度。如果为None，则计算所有元素的最大值并返回包含单个元素的Tensor变量，否则必须在  :math:`[-x.ndim, x.ndim]` 范围内。如果 :math:`axis[i] <0` ，则维度将变为 :math:`x.ndim+axis[i]` ，默认值为None。
   - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如果keepdim 为 False，结果张量的维度将比输入张量的小，默认值为False。
   - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
   Tensor, 在指定axis上进行求最大值运算的Tensor，数据类型和输入数据类型一致。


代码示例
::::::::::

..  code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()

    # data_x is a variable with shape [2, 4]
    # the axis is a int element
    x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                          [0.1, 0.2, 0.6, 0.7]])
    result1 = paddle.max(x)
    print(result1.numpy())
    #[0.9]
    result2 = paddle.max(x, axis=0)
    print(result2.numpy()) 
    #[0.2 0.3 0.6 0.9]
    result3 = paddle.max(x, axis=-1)
    print(result3.numpy())
    #[0.9 0.7]
    result4 = paddle.max(x, axis=1, keepdim=True)
    print(result4.numpy())
    #[[0.9]
    # [0.7]]

    # data_y is a variable with shape [2, 2, 2]
    # the axis is list 
    y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]])
    result5 = paddle.max(y, axis=[1, 2])
    print(result5.numpy())
    #[4. 8.]
    result6 = paddle.max(y, axis=[0, 1])
    print(result6.numpy())
    #[7. 8.]
