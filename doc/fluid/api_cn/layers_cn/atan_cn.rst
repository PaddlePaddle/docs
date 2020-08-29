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
    - x (Tensor) - 输入的Tensor，数据类型为：float16、float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()
        x_data = np.array([-0.8183,  0.4912, -0.6444,  0.0371]).astype(np.float32)
        x = paddle.to_variable(x_data)
        res = paddle.atan(x)
        print(res.numpy())
        # [-0.6858,  0.4566, -0.5724,  0.0371]
