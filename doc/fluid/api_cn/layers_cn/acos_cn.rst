.. _cn_api_fluid_layers_acos:

acos
-------------------------------

.. py:function:: paddle.fluid.layers.acos(x, name=None)

:alias_main: paddle.acos
:alias: paddle.acos,paddle.tensor.acos,paddle.tensor.math.acos
:old_api: paddle.fluid.layers.acos






arccosine激活函数。

.. math::
    out = cos^{-1}(x)

参数:
    - **x(Variable)** - acos的输入Tensor，数据类型为 float32 或 float64
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。
返回：  `acos` 的输出Tensor，数据类型与 `x` 相同。

返回类型： Variable



**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[4])
        # if data is [-0.8183,  0.4912, -0.6444,  0.0371]
        result = fluid.layers.acos(data)
        # result is [2.5293, 1.0573, 2.2711, 1.5336]



