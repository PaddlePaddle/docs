###################
 fluid.initializer
###################


.. _cn_api_fluid_initializer_Bilinear:

Bilinear
-------------------------------

.. py:attribute:: paddle.fluid.initializer.Bilinear

``BilinearInitializer`` 的别名


.. _cn_api_fluid_initializer_BilinearInitializer:

BilinearInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.BilinearInitializer

该初始化函数用于转置卷积函数，进行上采样。用户通过任意整型因子放大shape为(B，C，H，W)的特征图。用法如下：

**代码示例**:

.. code-block:: python

    factor = 2
    C = 2
    w_attr = fluid.initializer.ParamAttr(
        learning_rate=0.,
        regularizer=fluid.regularizer.L2Decay(0.),
        initializer=fluid.initializer.Bilinear())
    x = fluid.layers.data(name="data", shape=[3, 32, 32],
                          dtype="float32")
    conv_up = fluid.layers.conv2d_transpose(
        input=x,
        num_filters=C,
        output_size=None,
        filter_size=2 * factor - factor % 2,
        padding=int(math.ceil((factor - 1) / 2.)),
        stride=factor,
        groups=C,
        param_attr=w_attr,
        bias_attr=False)

num_filters = C和groups = C 表示这是按通道转置的卷积函数。滤波器shape为(C,1,K,K)，K为filter_size。该初始化函数为滤波器的每个通道设置(K,K)插值核。输出特征图的最终输出shape为(B,C,factor*H,factor*W)。注意学习率和权重衰减设为0，以便在训练过程中双线性插值的系数值保持不变





.. _cn_api_fluid_initializer_Constant:

Constant
-------------------------------

.. py:attribute:: paddle.fluid.initializer.Constant

``ConstantInitializer`` 的别名


.. _cn_api_fluid_initializer_ConstantInitializer:

ConstantInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False)

常量初始器

参数：
        - **value** (float) - 用常量初始化变量

**代码示例**

.. code-block:: python
        
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Constant(value=2.0))







.. _cn_api_fluid_initializer_force_init_on_cpu:

force_init_on_cpu
-------------------------------

.. py:function:: paddle.fluid.initializer.force_init_on_cpu()

标志位，是否强制在CPU上进行变量初始化。

返回：状态，是否应强制在CPU上强制进行变量初始化

返回类型：bool

**代码示例**：

.. code-block:: python

    if fluid.initializer.force_init_on_cpu():
        step = fluid.layers.create_global_var(shape=[2,3], value=1.0, dtype='float32')











.. _cn_api_fluid_initializer_init_on_cpu:

init_on_cpu
-------------------------------

.. py:function:: paddle.fluid.initializer.init_on_cpu()

强制变量在 cpu 上初始化。

**代码示例**

.. code-block:: python
        
        with fluid.initializer.init_on_cpu():
            step = fluid.layers.create_global_var(shape=[2,3], value=1.0, dtype='float32')






.. _cn_api_fluid_initializer_MSRA:

MSRA
-------------------------------

.. py:attribute:: paddle.fluid.initializer.MSRA

``MSRAInitializer`` 的别名

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

    x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    fc = fluid.layers.fc(input=x, size=10, param_attr=fluid.initializer.MSRA(uniform=False))






.. _cn_api_fluid_initializer_Normal:

Normal
-------------------------------

.. py:attribute:: paddle.fluid.initializer.Normal

``NormalInitializer`` 的别名


.. _cn_api_fluid_initializer_NormalInitializer:

NormalInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)

随机正态（高斯）分布初始化器

参数：
        - **loc** （float） - 正态分布的平均值
        - **scale** （float） - 正态分布的标准差
        - **seed** （int） - 随机种子

**代码示例**

.. code-block:: python

        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0)


.. _cn_api_fluid_initializer_NumpyArrayInitializer:

NumpyArrayInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.NumpyArrayInitializer(value)

使用Numpy型数组来初始化参数变量。

参数：
        - **value** （numpy） - 用于初始化变量的一个Numpy型数组。

**代码示例**

.. code-block:: python

    x = fluid.layers.data(name="x", shape=[5], dtype='float32')
    fc = fluid.layers.fc(input=x, size=10,
        param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))


.. _cn_api_fluid_initializer_TruncatedNormal:

TruncatedNormal
-------------------------------

.. py:attribute:: paddle.fluid.initializer.TruncatedNormal

``TruncatedNormalInitializer`` 的别名


.. _cn_api_fluid_initializer_TruncatedNormalInitializer:

TruncatedNormalInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=1.0, seed=0)

Random Truncated Normal（高斯）分布初始化器

参数：
        - **loc** （float） - 正态分布的平均值
        - **scale** （float） - 正态分布的标准差
        - **seed** （int） - 随机种子

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))









.. _cn_api_fluid_initializer_Uniform:

Uniform
-------------------------------

.. py:attribute:: paddle.fluid.initializer.Uniform

``UniformInitializer`` 的别名



.. _cn_api_fluid_initializer_UniformInitializer:

UniformInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0) 

随机均匀分布初始化器

参数：
        - **low** (float) - 下界 
        - **high** (float) - 上界
        - **seed** (int) - 随机种子

**代码示例**

.. code-block:: python
       
       import paddle.fluid as fluid
       x = fluid.layers.data(name='x', shape=[1], dtype='float32')
       fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5))
 








.. _cn_api_fluid_initializer_Xavier:

Xavier
-------------------------------

.. py:attribute:: paddle.fluid.initializer.Xavier

``XavierInitializer`` 的别名






.. _cn_api_fluid_initializer_XavierInitializer:

XavierInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.XavierInitializer(uniform=True, fan_in=None, fan_out=None, seed=0)

该类实现Xavier权重初始化方法（ Xavier weight initializer），Xavier权重初始化方法出自Xavier Glorot和Yoshua Bengio的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

该初始化函数用于保持所有层的梯度尺度几乎一致。

在均匀分布的情况下，取值范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in+fan\_out}}

正态分布的情况下，均值为0，标准差为：

.. math::
    
    x = \sqrt{\frac{2.0}{fan\_in+fan\_out}}

参数：
    - **uniform** (bool) - 是否用均匀分布或者正态分布
    - **fan_in** (float) - 用于Xavier初始化的fan_in。如果为None，fan_in沿伸自变量
    - **fan_out** (float) - 用于Xavier初始化的fan_out。如果为None，fan_out沿伸自变量
    - **seed** (int) - 随机种子

.. note::

    在大多数情况下推荐将fan_in和fan_out设置为None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    queries = fluid.layers.data(name='x', shape=[1], dtype='float32')
    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.Xavier(uniform=False))






