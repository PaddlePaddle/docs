.. _cn_api_tensor_random_rand:

rand
----------------------

.. py:function:: paddle.rand(shape, dtype=None, name=None)

该OP返回符合均匀分布的，范围在[0, 1)的Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
    - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。
    - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持float32、float64。当该参数值为None时， 输出Tensor的数据类型为float32。默认值为None.
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
    Tensor: 符合均匀分布的范围为[0, 1)的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    # example 1: attr shape is a list which doesn't contain Tensor.
    out1 = paddle.rand(shape=[2, 3])
    # [[0.451152  , 0.55825245, 0.403311  ],  # random
    #  [0.22550228, 0.22106001, 0.7877319 ]]  # random

    # example 2: attr shape is a list which contains Tensor.
    dim1 = paddle.full([1], 2, "int64")
    dim2 = paddle.full([1], 3, "int32")
    out2 = paddle.rand(shape=[dim1, dim2, 2])
    # [[[0.8879919 , 0.25788337],  # random
    #   [0.28826773, 0.9712097 ],  # random
    #   [0.26438272, 0.01796806]],  # random
    #  [[0.33633623, 0.28654453],  # random
    #   [0.79109055, 0.7305809 ],  # random
    #   [0.870881  , 0.2984597 ]]]  # random

    # example 3: attr shape is a Tensor, the data type must be int64 or int32.
    shape_tensor = paddle.to_tensor(np.array([2, 3]))
    out2 = paddle.rand(shape_tensor)
    # [[0.22920267, 0.841956  , 0.05981819],  # random
    #  [0.4836288 , 0.24573246, 0.7516129 ]]  # random
