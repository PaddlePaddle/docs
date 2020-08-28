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

抛出异常
::::::::::
  - ``TypeError`` - 如果 ``shape`` 的类型不是list、tuple、Tensor。
  - ``TypeError`` - 如果 ``dtype`` 不是float32、float64。

示例代码
::::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    # example 1: attr shape is a list which doesn't contain Tensor.
    result_1 = paddle.standard_normal(shape=[2, 3])
    # [[-2.923464  ,  0.11934398, -0.51249987],
    #  [ 0.39632758,  0.08177969,  0.2692008 ]]

    # example 2: attr shape is a list which contains Tensor.
    dim_1 = paddle.fill_constant([1], "int64", 2)
    dim_2 = paddle.fill_constant([1], "int32", 3)
    result_2 = paddle.standard_normal(shape=[dim_1, dim_2, 2])
    # [[[-2.8852394 , -0.25898588],
    #   [-0.47420555,  0.17683524],
    #   [-0.7989969 ,  0.00754541]],
    #  [[ 0.85201347,  0.32320443],
    #   [ 1.1399018 ,  0.48336947],
    #   [ 0.8086993 ,  0.6868893 ]]]

    # example 3: attr shape is a Tensor, the data type must be int64 or int32.
    var_shape = paddle.to_tensor(np.array([2, 3]))
    result_3 = paddle.standard_normal(var_shape)
    # [[-2.878077 ,  0.17099959,  0.05111201]
    #  [-0.3761474, -1.044801  ,  1.1870178 ]]
