.. _cn_api_fluid_disable_imperative:

disable_imperative
-------------------------------

.. py:function:: paddle.fluid.disable_imperative()

该接口退出动态图模式。

返回：无

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid

    fluid.enable_imperative()  # Now we are in imperative mode
    fluid.disable_imperative() # Now we are in declarative mode
