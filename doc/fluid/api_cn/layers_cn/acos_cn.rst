.. _cn_api_fluid_layers_acos:

acos
-------------------------------

.. py:function:: paddle.fluid.layers.acos(x, name=None)

:alias_main: paddle.acos
:alias: paddle.acos,paddle.tensor.acos,paddle.tensor.math.acos
:old_api: paddle.fluid.layers.acos



arccosine函数。

.. math::
    out = cos^{-1}(x)

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor


**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([-0.8183,  0.4912, -0.6444,  0.0371]).astype(np.float32)
        x = paddle.imperative.to_variable(x_data)
        res = paddle.acos(x)
        print(res.numpy())
        # [2.5293, 1.0573, 2.2711, 1.5336]
