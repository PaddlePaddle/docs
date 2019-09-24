.. _cn_api_fluid_layers_acos:

acos
-------------------------------

.. py:function:: paddle.fluid.layers.acos(x, name=None)

arccosine激活函数。

.. math::
    out = cos^{-1}(x)

参数:
    - **x(Variable)** - acos的输入，数据类型为 float32 或 float64
    - **name(None|str)** – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

返回：  `acos` 的输出，数据类型与 `x` 相同。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[4])
        # if data is [-0.8183,  0.4912, -0.6444,  0.0371]
        result = fluid.layers.acos(data)
        # result is [2.5293, 1.0573, 2.2711, 1.5336]



