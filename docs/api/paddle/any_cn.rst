.. _cn_api_tensor_any:

any
-------------------------------

.. py:function:: paddle.any(x, axis=None, keepdim=False, name=None)

对指定维度上的 Tensor 元素进行逻辑或运算，并输出相应的计算结果。

参数
:::::::::
    - **x** (Tensor) - 输入变量为多维 Tensor，数据类型为 bool。
    - **axis** (int|list|tuple，可选) - 计算逻辑或运算的维度。如果为 None，则计算所有元素的逻辑或并返回包含单个元素的 Tensor，否则必须在  :math:`[−rank(x),rank(x)]` 范围内。如果 :math:`axis [i] <0`，则维度将变为 :math:`rank+axis[i]`，默认值为 None。
    - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回
:::::::::
  Tensor，在指定维度上进行逻辑或运算的 Tensor，数据类型和输入数据类型一致。


代码示例
:::::::::

..  code-block:: python

    import paddle
    import numpy as np

    # x is a bool Tensor variable with following elements:
    #    [[True, False]
    #     [False, False]]
    x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
    print(x)
    x = paddle.cast(x, 'bool')

    # out1 should be [True]
    out1 = paddle.any(x)  # [True]
    print(out1)

    # out2 should be [True, True]
    out2 = paddle.any(x, axis=0)  # [True, True]
    print(out2)

    # keepdim=False, out3 should be [True, True], out.shape should be (2,)
    out3 = paddle.any(x, axis=-1)  # [True, True]
    print(out3)

    # keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
    out4 = paddle.any(x, axis=1, keepdim=True)
    out4 = paddle.cast(out4, 'int32')  # [[True], [True]]
    print(out4)
