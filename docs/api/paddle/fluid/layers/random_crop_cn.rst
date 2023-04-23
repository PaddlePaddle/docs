.. _cn_api_fluid_layers_random_crop:

random_crop
-------------------------------

.. py:function:: paddle.fluid.layers.random_crop(x, shape, seed=None)




该操作对batch中每个实例进行随机裁剪，即每个实例的裁剪位置不同，裁剪位置由均匀分布随机数生成器决定。所有裁剪后的实例都具有相同的维度，由 ``shape`` 参数决定。

参数
::::::::::::

    - **x(Variable)** - 多维Tensor。
    - **shape(list(int))** - 裁剪后最后几维的形状，注意，``shape`` 的个数小于 ``x`` 的秩。
    - **seed(int|Variable，可选)** - 设置随机数种子，默认情况下，种子是[-65536,-65536)中一个随机数，如果类型是Variable，要求数据类型是int64，默认值：None。

返回
::::::::::::
 裁剪后的Tensor。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.random_crop