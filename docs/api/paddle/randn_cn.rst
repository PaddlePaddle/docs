.. _cn_api_tensor_random_randn:

randn
-------------------------------

.. py:function:: paddle.randn(shape, dtype=None, name=None)

返回符合标准正态分布（均值为0，标准差为1的正态随机分布）的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
  - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是 int，或者是形状为[1]且数据类型为 int32、int64 的 Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的1-D Tensor。
  - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float32、float64。当该参数值为 None 时，输出 Tensor 的数据类型为 float32。默认值为 None。
  - **name** (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见  :ref:`api_guide_Name`。

返回
::::::::::
  Tensor：符合标准正态分布的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

.. code-block:: python

    import paddle

    # example 1: attr shape is a list which doesn't contain Tensor.
    out1 = paddle.randn(shape=[2, 3])
    # [[-2.923464  ,  0.11934398, -0.51249987],  # random
    #  [ 0.39632758,  0.08177969,  0.2692008 ]]  # random

    # example 2: attr shape is a list which contains Tensor.
    dim1 = paddle.to_tensor([2], 'int64')
    dim2 = paddle.to_tensor([3], 'int32')
    out2 = paddle.randn(shape=[dim1, dim2, 2])
    # [[[-2.8852394 , -0.25898588],  # random
    #   [-0.47420555,  0.17683524],  # random
    #   [-0.7989969 ,  0.00754541]],  # random
    #  [[ 0.85201347,  0.32320443],  # random
    #   [ 1.1399018 ,  0.48336947],  # random
    #   [ 0.8086993 ,  0.6868893 ]]]  # random

    # example 3: attr shape is a Tensor, the data type must be int64 or int32.
    shape_tensor = paddle.to_tensor([2, 3])
    out3 = paddle.randn(shape_tensor)
    # [[-2.878077 ,  0.17099959,  0.05111201]  # random
    #  [-0.3761474, -1.044801  ,  1.1870178 ]]  # random
