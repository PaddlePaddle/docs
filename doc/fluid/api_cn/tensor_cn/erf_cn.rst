.. _cn_api_tensor_erf:

erf
-------------------------------

.. py:function:: paddle.erf(x, name=None)

:alias_main: paddle.erf
:alias: paddle.erf,paddle.tensor.erf,paddle.tensor.math.erf,paddle.nn.functional.erf,paddle.nn.functional.activation.erf
:old_api: paddle.fluid.layers.erf



逐元素计算 Erf 激活函数。更多细节请参考 `Error function <https://en.wikipedia.org/wiki/Error_function>`_ 。


.. math::
    out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

参数：
    - **x** (Variable) - Erf Op 的输入，多维 Tensor 或 LoDTensor，数据类型为 float16, float32 或 float64。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
  - 多维 Tensor, 数据类型为 float16, float32 或 float64， 和输入 x 的数据类型相同，形状和输入 x 相同。

返回类型：
  - Tensor

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x_data = np.array([-0.4, -0.2, 0.1, 0.3])
    x = paddle.to_variable(x_data)
    out = paddle.erf(x)
    print(out.numpy())
    # [-0.42839236 -0.22270259  0.11246292  0.32862676]
