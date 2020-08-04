.. _cn_api_fluid_layers_ceil:

ceil
-------------------------------

.. py:function:: paddle.fluid.layers.ceil(x, name=None)

:alias_main: paddle.ceil
:alias: paddle.ceil,paddle.tensor.ceil,paddle.tensor.math.ceil
:old_api: paddle.fluid.layers.ceil



向上取整运算函数。

.. math::
    out = \left \lceil x \right \rceil



参数:
    - **x** (Tensor) - 该OP的输入为多维Tensor。数据类型为float32或float64。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回： 输出为Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

        import paddle
        import numpy as np

        paddle.enable_imperative()
        x_data = np.array([[-1.5,6],[1,15.6]]).astype(np.float32)
        x = paddle.imperative.to_variable(x_data)
        res = paddle.ceil(x)
        print(res.numpy())
        # [[-1.  6.]
        # [ 1. 16.]]
