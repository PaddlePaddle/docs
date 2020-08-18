.. _cn_api_fluid_dygraph_disable_imperative:

disable_imperative
-------------------------------

.. py:function:: paddle.fluid.dygraph.disable_imperative()

该接口退出动态图模式。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid

    fluid.dygraph.enable_imperative()  # Now we are in imperative mode
    x = fluid.layers.ones( (2, 2), "float32")
    y = fluid.layers.zeros( (2, 2), "float32")
    z = x + y
    print( z.numpy() )   #[[1, 1], [1, 1]]
    fluid.dygraph.disable_imperative() # Now we are in declarative mode
