.. _cn_api_fluid_initializer_force_init_on_cpu:

force_init_on_cpu
-------------------------------

.. py:function:: paddle.fluid.initializer.force_init_on_cpu()

标志位，是否强制在CPU上进行变量初始化。

返回：状态，是否应强制在CPU上强制进行变量初始化

返回类型：bool

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    if fluid.initializer.force_init_on_cpu():
        step = fluid.layers.create_global_var(shape=[2,3], value=1.0, dtype='float32')











