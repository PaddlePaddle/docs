.. _cn_api_fluid_initializer_ConstantInitializer:

ConstantInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.ConstantInitializer(value=0.0, force_cpu=False)

该接口为常量初始化函数，用于权重初始化，通过输入的value值初始化输入变量；

参数：
        - **value** (float) - 用于初始化输入变量的值；

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.ConstantInitializer(value=2.0))







