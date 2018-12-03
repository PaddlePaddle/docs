
#################
fluid.initializer
#################


.. _cn_api_fluid_initializer_BilinearInitializer:

BilinearInitializer
>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.BilinearInitializer

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


.. _cn_api_fluid_initializer_ConstantInitializer:

ConstantInitializer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False)

常量初始器

参数：
        - **value** (float) - 用常量初始化变量

**代码示例**

.. code-block:: python
        
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Constant(value=2.0))

.. _cn_api_fluid_initializer_init_on_cpu:

init_on_cpu
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.init_on_cpu(*args, **kwds)

强制变量在 cpu 上初始化。

**代码示例**

.. code-block:: python
        
        with init_on_cpu():
                step = layers.create_global_var()

.. _cn_api_fluid_initializer_NormalInitializer:

NormalInitializer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)

随机正态（高斯）分布初始化器

参数：
        - **loc** （float） - 正态分布的平均值
        - **scale** （float） - 正态分布的标准差
        - **seed** （int） - 随机种子

**代码示例**

.. code-block:: python

        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0)



.. _cn_api_fluid_initializer_TruncatedNormalInitializer:

TruncatedNormalInitializer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=1.0, seed=0)

Random Truncated Normal（高斯）分布初始化器

参数：
        - **loc** （float） - 正态分布的平均值
        - **scale** （float） - 正态分布的标准差
        - **seed** （int） - 随机种子

**代码示例**

.. code-block:: python

        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))




.. _cn_api_fluid_initializer_UniformInitializer:

UniformInitializer
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0) 

随机均匀分布初始化器

参数：
        - **low** (float) - 下界 
        - **high** (float) - 上界
        - **seed** (float) - 随机种子

**代码示例**

.. code-block:: python
        
       fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5))
 

