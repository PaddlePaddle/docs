.. _cn_api_tensor_random_randint:

randint
-------------------------------

.. py:function:: paddle.randint(low=0, high=None, shape=[1], dtype=None, name=None)

该OP返回服从均匀分布的、范围在[``low``, ``high``)的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。当 ``high`` 为None时（默认），均匀采样的区间为[0, ``low``)。

参数
::::::::::
    - **low** (int) - 要生成的随机值范围的下限，``low`` 包含在范围中。当 ``high`` 为None时，均匀采样的区间为[0, ``low``)。默认值为0。
    - **high** (int, 可选) - 要生成的随机值范围的上限，``high`` 不包含在范围中。默认值为None，此时范围是[0, ``low``)。
    - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。。默认值为[1]。
    - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持int32、int64。当该参数值为None时， 输出Tensor的数据类型为int64。默认值为None.
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
    Tensor：从区间[``low``，``high``)内均匀分布采样的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

代码示例
:::::::::::

.. code-block:: python

    import paddle

    # example 1:
    # attr shape is a list which doesn't contain Tensor.
    out1 = paddle.randint(low=-5, high=5, shape=[3])
    # [0, -3, 2]  # random

    # example 2:
    # attr shape is a list which contains Tensor.
    dim1 = paddle.to_tensor([2], 'int64')
    dim2 = paddle.to_tensor([3], 'int32')
    out2 = paddle.randint(low=-5, high=5, shape=[dim1, dim2])
    # [[0, -1, -3],  # random
    #  [4, -2,  0]]  # random

    # example 3:
    # attr shape is a Tensor
    shape_tensor = paddle.to_tensor(3)
    out3 = paddle.randint(low=-5, high=5, shape=shape_tensor)
    # [-2, 2, 3]  # random

    # example 4:
    # data type is int32
    out4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')
    # [-5, 4, -4]  # random

    # example 5:
    # Input only one parameter
    # low=0, high=10, shape=[1], dtype='int64'
    out5 = paddle.randint(10)
    # [7]  # random
