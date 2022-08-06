.. _cn_api_fluid_layers_uniform_random:

uniform_random
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0, name=None)




该OP返回数值服从范围[``min``, ``max``)内均匀分布的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

::

    示例1:
             给定：
                 shape=[1,2]
             则输出为：
                 result=[[0.8505902, 0.8397286]]

参数
::::::::::::

    - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。
    - **dtype** (str|np.dtype|core.VarDesc.VarType，可选) - 输出Tensor的数据类型，支持float32、float64。默认值为float32。
    - **min** (float|int，可选) - 要生成的随机值范围的下限，min包含在范围中。支持的数据类型：float、int。默认值为-1.0。
    - **max** (float|int，可选) - 要生成的随机值范围的上限，max不包含在范围中。支持的数据类型：float、int。默认值为1.0。
    - **seed** (int，可选) - 随机种子，用于生成样本。0表示使用系统生成的种子。注意如果种子不为0，该操作符每次都生成同样的随机数。支持的数据类型：int。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tensor：数值服从范围[``min``, ``max``)内均匀分布的随机Tensor，形状为 ``shape``，数据类型为 ``dtype``。

抛出异常
::::::::::::

    - ``TypeError`` - 如果 ``shape`` 的类型不是list、tuple、Tensor。
    - ``TypeError`` - 如果 ``dtype`` 不是float32、float64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.uniform_random