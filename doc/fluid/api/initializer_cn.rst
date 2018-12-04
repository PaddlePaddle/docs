.. _cn_api_fluid_initializer_BilinearInitializer:

BilinearInitializer
>>>>>>>>>>>>>>>>>>>>>

.. py:class:: class paddle.fluid.initializer.BilinearInitializer

该初始化函数用于转置卷积函数，进行上采样。用户通过任意整型因子放大shape为(B，C，H，W)的特征图。用法如下：

**代码示例**:

.. code-block:: python

    factor = 2
    w_attr = ParamAttr(learning_rate=0., regularizer=L2Decay(0.),
                   initializer=Bilinear())
    conv_up = fluid.layers.conv2d_transpose(
        input,
        num_filters=C,
        output_size=None,
        filter_size=2 * factor - factor % 2,
        padding=ceil((factor - 1) / 2.),
        stride=factor,
        groups=C,
        param_attr=w_attr,
        bias_attr=False)

num_filters = C和groups = C 表示这是按通道转置的卷积函数。滤波器shape为(C,1,K,K)，K为filter_size。该初始化函数为滤波器的每个通道设置(K,K)插值核。输出特征图的最终输出shape为(B,C,factor*H,factor*W)注意学习率和权重衰减设为0，以便在训练过程中双线性插值的系数值保持不变

.. _cn_api_fluid_initializer_XavierInitializer:

XavierInitializer
>>>>>>>>>>>>>>>>>>>

.. py:class:: class paddle.fluid.initializer.XavierInitializer(uniform=True, fan_in=None, fan_out=None, seed=0)

该类实现Xavier权重初始化方法（ Xavier weight initializer），Xavier权重初始化方法出自Xavier Glorot和Yoshua Bengio的论文 Understanding the difficulty of training deep feedforward neural networks

该初始化函数用于使所有层的梯度尺度几乎保持一致。在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in+fan\_out}}

在正态分布中，均值为0，标准差为：

.. math::

    x = \sqrt{\frac{2.0}{fan\_in+fan\_out}}

参数：
    - **uniform** (bool) - 是否用均匀分布或者正态分布
    - **fan_in** (float) - 用于Xavier初始化的fan_in。如果为None，fan_in则沿伸自变量
    - **fan_out** (float) - 用于Xavier初始化的fan_out。如果为None，fan_out则沿伸自变量
    - **seed** (int) - 随机种子

**注解**：
    在大多数情况下推荐将fan_in和fan_out设置为None

**代码示例**：

.. code-block:: python

    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.Xavier(uniform=False))

.. _cn_api_fluid_initializer_MSRAInitializer:

MSRAInitializer
>>>>>>>>>>>>>>>>>

.. py:class:: class paddle.fluid.initializer.MSRAInitializer(uniform=True, fan_in=None, seed=0)

实现MSRA初始化函数（a.k.a. Kaiming初始化函数）

该类实现权重初始化方法，方法来自Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun所写的论文 : _Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification: https://arxiv.org/abs/1502.01852。这是一个鲁棒性特别强的初始化方法，特别数考虑到非线性纠正激活函数（rectifier nonlinearities）。在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in}}

在正态分布中，均值为0，标准差为：

.. math::

    \sqrt{\frac{2.0}{fan\_in}}

参数：
    - **uniform** (bool) - 是否用均匀分布或正态分布
    - **fan_in** (float) - MSRAInitializer的fan_in。如果为None，fan_in沿伸自变量
    - **seed** (int) - 随机种子

**注解**
    在大多数情况下推荐设置fan_in为None

**代码示例**：

.. code-block:: python

    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.MSRA(uniform=False))


