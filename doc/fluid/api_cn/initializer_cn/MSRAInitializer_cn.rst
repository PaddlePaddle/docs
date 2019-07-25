.. _cn_api_fluid_initializer_MSRAInitializer:

MSRAInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.MSRAInitializer(uniform=True, fan_in=None, seed=0)

实现MSRA初始化（a.k.a. Kaiming初始化）

该类实现权重初始化方法，方法来自Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun所写的论文: `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 。这是一个鲁棒性特别强的初始化方法，并且适应了非线性激活函数（rectifier nonlinearities）。

在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in}}

在正态分布中，均值为0，标准差为：

.. math::

    \sqrt{\frac{2.0}{fan\_in}}

参数：
    - **uniform** (bool) - 是否用均匀分布或正态分布
    - **fan_in** (float) - MSRAInitializer的fan_in。如果为None，fan_in沿伸自变量
    - **seed** (int) - 随机种子

.. note:: 

    在大多数情况下推荐设置fan_in为None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    fc = fluid.layers.fc(input=x, size=10, param_attr=fluid.initializer.MSRA(uniform=False))






