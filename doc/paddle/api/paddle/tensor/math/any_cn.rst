.. _cn_api_tensor_any:

any
-------------------------------

.. py:function:: paddle.any(x, axis=None, keepdim=False, name=None)

该OP是对指定维度上的Tensor元素进行逻辑或运算，并输出相应的计算结果。

参数：
    - **x** （Tensor）- 输入变量为多维Tensor，数据类型为bool。
    - **axis** （int | list | tuple ，可选）- 计算逻辑或运算的维度。如果为None，则计算所有元素的逻辑或并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(x),rank(x)]` 范围内。如果 :math:`axis [i] <0` ，则维度将变为 :math:`rank+axis[i]` ，默认值为None。
    - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如 keepdim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
  ``Tensor``，在指定维度上进行逻辑或运算的Tensor，数据类型和输入数据类型一致。


**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    import numpy as np

    # set as static mode
    paddle.disable_static()

    # x is a bool Tensor variable with following elements:
    #    [[True, False]
    #     [False, False]]
    x = layers.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
    print(x)
    x = layers.cast(x, 'bool')

    # out1 should be [True]
    out1 = paddle.any(x)  # [True]
    print(out1)

    # out2 should be [True, False]
    out2 = paddle.any(x, axis=0)  # [True, False]
    print(out2)

    # keep_dim=False, out3 should be [True, False], out.shape should be (2,)
    out3 = paddle.any(x, axis=-1)  # [True, False]
    print(out3)

    # keep_dim=True, result should be [[True], [False]], out.shape should be (2,1)
    out4 = paddle.any(x, axis=1, keep_dim=True)
    out4 = layers.cast(out4, 'int32')  # [[True], [False]]
    print(out4)
