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

        import paddle.fluid as fluid
        x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
        fc = fluid.layers.fc(input=x, size=10,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0)


