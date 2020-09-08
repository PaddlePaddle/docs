.. _cn_api_tensor_uniform:

uniform
-------------------------------

.. py:function:: paddle.uniform(shape, dtype='float32', min=-1.0, max=1.0, seed=0, name=None)




该OP返回数值服从范围[``min``, ``max``)内均匀分布的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

::

    示例1:
             给定：
                 shape=[1,2]
             则输出为：
                 result=[[0.8505902, 0.8397286]]

参数：
    - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。
    - **dtype** (str|np.dtype， 可选) - 输出Tensor的数据类型，支持float32、float64。默认值为float32。
    - **min** (float|int，可选) - 要生成的随机值范围的下限，min包含在范围中。支持的数据类型：float、int。默认值为-1.0。
    - **max** (float|int，可选) - 要生成的随机值范围的上限，max不包含在范围中。支持的数据类型：float、int。默认值为1.0。
    - **seed** (int，可选) - 随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。支持的数据类型：int。默认为 0。
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：
    Tensor：数值服从范围[``min``, ``max``)内均匀分布的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

抛出异常：
    - ``TypeError`` - 如果 ``shape`` 的类型不是list、tuple、Tensor。
    - ``TypeError`` - 如果 ``dtype`` 不是float32、float64。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()

    # example 1:
    # attr shape is a list which doesn't contain Tensor.
    result_1 = paddle.uniform(shape=[3, 4])
    # [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357],
    #  [-0.34646994, -0.45116323, -0.09902662, -0.11397249],
    #  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]]

    # example 2:
    # attr shape is a list which contains Tensor.
    dim_1 = paddle.fill_constant([1], "int64", 2)
    dim_2 = paddle.fill_constant([1], "int32", 3)
    result_2 = paddle.uniform(shape=[dim_1, dim_2])
    # [[-0.9951253,   0.30757582, 0.9899647 ],
    #  [ 0.5864527,   0.6607096,  -0.8886161 ]]

    # example 3:
    # attr shape is a Tensor, the data type must be int64 or int32.
    shape = np.array([2, 3])
    shape_tensor = paddle.to_tensor(shape)

    result_3 = paddle.uniform(shape_tensor)
    # if shape_tensor's value is [2, 3]
    # result_3 is:
    # [[-0.8517412,  -0.4006908,   0.2551912 ],
    #  [ 0.3364414,   0.36278176, -0.16085452]]








