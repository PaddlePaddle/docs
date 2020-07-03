.. _cn_api_fluid_enable_imperative:

enable_imperative
-------------------------------

.. py:function:: paddle.fluid.enable_imperative(place=None)

该接口打开动态图模式。

参数
::::::::::::

  - **place** (fluid.CPUPlace 或 fluid.CUDAPlace，可选) - 执行动态图的设备。若为None，则设备根据paddle的编译方式决定。默认值为 ``None``。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid

    fluid.enable_imperative()  # Now we are in imperative mode
    x = fluid.layers.ones( (2, 2), "float32")
    y = fluid.layers.zeros( (2, 2), "float32")
    z = x + y
    print( z.numpy() )   #[[1, 1], [1, 1]]

