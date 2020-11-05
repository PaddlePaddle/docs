.. _cn_api_tensor_random_standard_normal:

standard_normal
-------------------------------

.. py:function:: paddle.standard_normal(shape, dtype=None, name=None)

该OP返回符合标准正态分布（均值为0，标准差为1的正态随机分布）的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
  - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。
  - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持float32、float64。当该参数值为None时， 输出Tensor的数据类型为float32。默认值为None.
  - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回
::::::::::
  Tensor：符合标准正态分布的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

.. code-block:: python

    import paddle

    # example 1: attr shape is a list which doesn't contain Tensor.
    out1 = paddle.standard_normal(shape=[2, 3])
    # [[-2.923464  ,  0.11934398, -0.51249987],  # random
    #  [ 0.39632758,  0.08177969,  0.2692008 ]]  # random

    # example 2: attr shape is a list which contains Tensor.
    dim1 = paddle.full([1], 2, "int64")
    dim2 = paddle.full([1], 3, "int32")
    out2 = paddle.standard_normal(shape=[dim1, dim2, 2])
    # [[[-2.8852394 , -0.25898588],  # random
    #   [-0.47420555,  0.17683524],  # random
    #   [-0.7989969 ,  0.00754541]],  # random
    #  [[ 0.85201347,  0.32320443],  # random
    #   [ 1.1399018 ,  0.48336947],  # random
    #   [ 0.8086993 ,  0.6868893 ]]]  # random

    # example 3: attr shape is a Tensor, the data type must be int64 or int32.
    shape_tensor = paddle.to_tensor(np.array([2, 3]))
    out3 = paddle.standard_normal(shape_tensor)
    # [[-2.878077 ,  0.17099959,  0.05111201]  # random
    #  [-0.3761474, -1.044801  ,  1.1870178 ]]  # random
