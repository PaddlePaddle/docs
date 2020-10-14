.. _cn_api_tensor_random_normal:

normal
-------------------------------

.. py:function:: paddle.normal(mean=0.0, std=1.0, shape=None, name=None)


该OP返回符合正态分布（均值为 ``mean`` ，标准差为 ``std`` 的正态随机分布）的随机Tensor。

如果 ``mean`` 是Tensor，则输出Tensor和 ``mean`` 具有相同的形状和数据类型。
如果 ``mean`` 不是Tensor，且 ``std`` 是Tensor，则输出Tensor和 ``std`` 具有相同的形状和数据类型。
如果 ``mean`` 和 ``std`` 都不是Tensor，则输出Tensor的形状为 ``shape``，数据类型为float32。

如果 ``mean`` 和 ``std`` 都是Tensor，则 ``mean`` 和 ``std`` 的元素个数应该相同。

参数
::::::::::
    - mean (float|Tensor, 可选) - 输出Tensor的正态分布的平均值。如果 ``mean`` 是float，则表示输出Tensor中所有元素的正态分布的平均值。如果 ``mean`` 是Tensor(支持的数据类型为float32、float64)，则表示输出Tensor中每个元素的正态分布的平均值。默认值为0.0
    - std (float|Tensor, 可选) - 输出Tensor的正态分布的标准差。如果 ``std`` 是float，则表示输出Tensor中所有元素的正态分布的标准差。如果 ``std`` 是Tensor(支持的数据类型为float32、float64)，则表示输出Tensor中每个元素的正态分布的标准差。默认值为0.0
    - shape (list|tuple|Tensor, 可选) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。如果 ``mean`` 或者 ``std`` 是Tensor，输出Tensor的形状和 ``mean`` 或者 ``std`` 相同(此时 ``shape`` 无效)。默认值为None。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
  Tensor：符合正态分布（均值为 ``mean`` ，标准差为 ``std`` 的正态随机分布）的随机Tensor。

示例代码
::::::::::

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()

    out1 = paddle.normal(shape=[2, 3])
    # [[ 0.17501129  0.32364586  1.561118  ]  # random
    #  [-1.7232178   1.1545963  -0.76156676]]  # random

    mean_tensor = paddle.to_tensor(np.array([1.0, 2.0, 3.0]))
    out2 = paddle.normal(mean=mean_tensor)
    # [ 0.18644847 -1.19434458  3.93694787]  # random

    std_tensor = paddle.to_tensor(np.array([1.0, 2.0, 3.0]))
    out3 = paddle.normal(mean=mean_tensor, std=std_tensor)
    # [1.00780561 3.78457445 5.81058198]  # random
