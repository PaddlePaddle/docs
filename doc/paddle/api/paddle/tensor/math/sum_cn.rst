.. _cn_api_tensor_sum:

sum
-------------------------------

.. py:function:: paddle.sum(x, axis=None, dtype=None, keepdim=False, name=None)

该OP是对指定维度上的Tensor元素进行求和运算，并输出相应的计算结果。

参数：
    - **x** （Tensor）- 输入变量为多维Tensor，支持数据类型为float32，float64，int32，int64。
    - **axis** （int | list | tuple ，可选）- 求和运算的维度。如果为None，则计算所有元素的和并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(x),rank(x)]` 范围内。如果 :math:`axis [i] <0` ，则维度将变为 :math:`rank+axis[i]` ，默认值为None。
    - **dtype** （str ， 可选）- 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为None。
    - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如 keepdim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
  ``Tensor``，在指定维度上进行求和运算的Tensor，数据类型和输入数据类型一致。


**代码示例**

..  code-block:: python

    import numpy as np
    import paddle

    # x is a Tensor variable with following elements:
    #    [[0.2, 0.3, 0.5, 0.9]
    #     [0.1, 0.2, 0.6, 0.7]]
    # Each example is followed by the corresponding output tensor.
    x_data = np.array([[0.2, 0.3, 0.5, 0.9],[0.1, 0.2, 0.6, 0.7]]).astype('float32')
    x = paddle.to_tensor(x_data)
    out1 = paddle.sum(x)  # [3.5]
    out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
    out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
    out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

    # y is a Tensor variable with shape [2, 2, 2] and elements as below:
    #      [[[1, 2], [3, 4]],
    #      [[5, 6], [7, 8]]]
    # Each example is followed by the corresponding output tensor.
    y_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype('float32')
    y = paddle.to_tensor(y_data)
    out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
    out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]
