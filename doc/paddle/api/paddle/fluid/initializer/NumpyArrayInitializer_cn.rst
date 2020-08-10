.. _cn_api_fluid_initializer_NumpyArrayInitializer:

NumpyArrayInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.NumpyArrayInitializer(value)




该OP使用Numpy型数组来初始化参数变量。

参数：
        - **value** （numpy） - 用于初始化变量的一个Numpy型数组。

返回：张量（Tensor）

返回类型：变量（Variable）

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy
    x1 = fluid.data(name="x1", shape=[2, 1], dtype='float32')
    fc = fluid.layers.fc(input=x1, size=10,
        param_attr=fluid.initializer.NumpyArrayInitializer(numpy.array([1,2])))


