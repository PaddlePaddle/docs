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
 








