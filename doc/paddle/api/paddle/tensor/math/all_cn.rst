.. _cn_api_tensor_all:

all
-------------------------------

.. py:function:: paddle.all(x, axis=None, keepdim=False, name=None)

该OP是对指定维度上的Tensor元素进行逻辑与运算，并输出相应的计算结果。

参数
:::::::::
    - **x** （Tensor）- 输入变量为多维Tensor，数据类型为bool。
    - **axis** （int | list | tuple ，可选）- 计算逻辑与运算的维度。如果为None，则计算所有元素的逻辑与并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(x),rank(x)]` 范围内。如果 :math:`axis [i] <0` ，则维度将变为 :math:`rank+axis[i]` ，默认值为None。
    - **keepdim** （bool）- 是否在输出Tensor中保留减小的维度。如 keepdim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
  Tensor，在指定维度上进行逻辑与运算的Tensor，数据类型和输入数据类型一致。


代码示例
:::::::::

..  code-block:: python

    import paddle
    import numpy as np

    # x is a bool Tensor variable with following elements:
    #    [[True, False]
    #     [True, True]]
    x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
    print(x)
    x = paddle.cast(x, 'bool')

    # out1 should be [False]
    out1 = paddle.all(x)  # [False]
    print(out1)

    # out2 should be [True, False]
    out2 = paddle.all(x, axis=0)  # [True, False]
    print(out2)

    # keepdim=False, out3 should be [False, True], out.shape should be (2,)
    out3 = paddle.all(x, axis=-1)  # [False, True]
    print(out3)

    # keepdim=True, out4 should be [[False], [True]], out.shape should be (2,1)
    out4 = paddle.all(x, axis=1, keepdim=True)
    out4 = paddle.cast(out4, 'int32')  # [[False], [True]]
    print(out4)
