

.. _cn_api_fluid_initializer_init_on_cpu:

init_on_cpu
>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.init_on_cpu(*args, **kwds)

强制变量在 cpu 上初始化。

**代码示例**

.. code-block:: python
        
        with init_on_cpu():
                step = layers.create_global_var()

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
 

.. _cn_api_fluid_initializer_NormalInitializer:

NormalInitializer
>>>>>>>>>>>>

.. py:class:: class paddle.fluid.initializer.NormalInitializer(loc=0.0, scale=1.0, seed=0)

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

Random TruncatedNormal（高斯）分布初始化器

参数：
        - **loc** （float） - 正态分布的平均值
        - **scale** （float） - 正态分布的标准差
        - **seed** （int） - 随机种子

**代码示例**

.. code-block:: python

        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=2.0))


