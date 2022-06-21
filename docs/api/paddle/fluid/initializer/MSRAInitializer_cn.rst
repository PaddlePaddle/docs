.. _cn_api_fluid_initializer_MSRAInitializer:

MSRAInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.MSRAInitializer(uniform=True, fan_in=None, seed=0, negative_slope=0.0, nonlinearity='relu')




该接口实现MSRA方式的权重初始化（a.k.a. Kaiming初始化）

该接口为权重初始化函数，方法来自Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun所写的论文: `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 。这是一个鲁棒性特别强的初始化方法，并且适应了非线性激活函数（rectifier nonlinearities）。
可以选择使用均匀分布或者正态分布初始化权重；
在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = gain \times \sqrt{\frac{3}{fan\_in}}

在正态分布中，均值为0，标准差为：

.. math::

    \frac{gain}{\sqrt{{fan\_in}}}

参数
::::::::::::

    - **uniform** (bool，可选) - 为True表示使用均匀分布，为False表示使用正态分布
    - **fan_in** (float16|float32，可选) - 可训练的Tensor的in_features值。如果设置为 None，程序会自动计算该值。如果你不想使用in_features，你可以自己设置这个值。默认值为None。
    - **seed** (int32，可选) - 随机种子
    - **negative_slope** (float，可选): 只适用于使用leaky_relu作为激活函数时的negative_slope参数。默认值为0.0。
    - **nonlinearity** (str，可选): 非线性激活函数。默认值为relu.

返回
::::::::::::
对象

.. note:: 

    在大多数情况下推荐设置fan_in为None

代码示例
::::::::::::

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    paddle.enable_static()
    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    fc = fluid.layers.fc(input=x, size=10, param_attr=fluid.initializer.MSRAInitializer(uniform=False))





