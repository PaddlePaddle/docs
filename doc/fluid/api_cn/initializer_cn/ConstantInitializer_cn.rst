.. _cn_api_fluid_initializer_ConstantInitializer:

ConstantInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False)

常量初始器

参数：
        - **value** (float) - 用常量初始化变量

**代码示例**

.. code-block:: python
        
        import paddle.fluid as fluid
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Constant(value=2.0))







