.. _cn_api_fluid_layers_asin:

asin
-------------------------------

.. py:function:: paddle.fluid.layers.asin(x, name=None)

:alias_main: paddle.asin
:alias: paddle.asin,paddle.tensor.asin,paddle.tensor.math.asin
:old_api: paddle.fluid.layers.asin



arcsine激活函数。

.. math::
    out = sin^{-1}(x)


参数:
    - **x(Variable)** - asin的输入Tensor，数据类型为 float32 或 float64
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。

返回：  `asin` 的输出Tensor，数据类型与 `x` 相同。

返回类型： Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[4])
        # if data is [-0.8183,  0.4912, -0.6444,  0.0371]
        result = fluid.layers.asin(data)
        # result is [-0.9585,  0.5135, -0.7003,  0.0372]



