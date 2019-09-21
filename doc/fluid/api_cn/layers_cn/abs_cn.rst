.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.fluid.layers.abs(x, name=None)

绝对值激活函数。

.. math::
    out = |x|

参数:

    - **x** - abs算子的输入
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：        abs算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.abs(data)


