.. _cn_api_fluid_initializer_NumpyArrayInitializer:

NumpyArrayInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.NumpyArrayInitializer(value)

使用Numpy型数组来初始化参数变量。

参数：
        - **value** （numpy） - 用于初始化变量的一个Numpy型数组。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[5], dtype='float32')
    fc = fluid.layers.fc(input=x, size=10,
        param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))


