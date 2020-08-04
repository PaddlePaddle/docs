.. _cn_api_fluid_layers_abs:

abs
-------------------------------

.. py:function:: paddle.fluid.layers.abs(x, name=None)

:alias_main: paddle.abs
:alias: paddle.abs,paddle.tensor.abs,paddle.tensor.math.abs
:old_api: paddle.fluid.layers.abs



绝对值函数。

.. math::
    out = |x|

参数:
    - **x** (Tensor)- 多维Tensor，数据类型为float32或float64。
    - **name** (str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：表示绝对值结果的Tensor，数据类型与 `x` 相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([-1, -2, -3, -4]).astype(np.float32)
        x = paddle.imperative.to_variable(x_data)
        res = paddle.abs(x)
        print(res.numpy())
