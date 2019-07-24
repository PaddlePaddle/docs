.. _cn_api_fluid_initializer_init_on_cpu:

init_on_cpu
-------------------------------

.. py:function:: paddle.fluid.initializer.init_on_cpu()

强制变量在 cpu 上初始化。

**代码示例**

.. code-block:: python
        
        import paddle.fluid as fluid
        with fluid.initializer.init_on_cpu():
            step = fluid.layers.create_global_var(shape=[2,3], value=1.0, dtype='float32')






