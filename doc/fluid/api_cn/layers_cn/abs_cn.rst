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
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

        import paddle
        paddle.disable_static()

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.abs(x)
        print(out.numpy())
        # [0.4 0.2 0.1 0.3]
