.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.fluid.layers.abs(x, name=None)

:alias_main: paddle.abs
:alias: paddle.abs,paddle.tensor.abs,paddle.tensor.math.abs
:old_api: paddle.fluid.layers.abs



绝对值激活函数。

.. math::
    out = |x|

参数:
    - **x** (Variable)- 多维Tensor，数据类型为float32或float64。
    - **name** (str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：表示绝对值结果的Tensor，数据类型与x相同。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.abs(data)
