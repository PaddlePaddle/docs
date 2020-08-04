.. _cn_api_fluid_layers_atan:

atan
-------------------------------

.. py:function:: paddle.fluid.layers.atan(x, name=None)

:alias_main: paddle.atan
:alias: paddle.atan,paddle.tensor.atan,paddle.tensor.math.atan
:update_api: paddle.fluid.layers.atan



arctangent函数。

.. math::
    out = tan^{-1}(x)

参数:
    - **x(Tensor)** - atan的输入Tensor，数据类型为 float32 或 float64
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。

返回：  `atan` 的输出Tensor，数据类型与 `x` 相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([-0.8183,  0.4912, -0.6444,  0.0371]).astype(np.float32)
        x = paddle.imperative.to_variable(x_data)
        res = paddle.atan(x)
        print(res.numpy())
        # [-0.6858,  0.4566, -0.5724,  0.0371]
