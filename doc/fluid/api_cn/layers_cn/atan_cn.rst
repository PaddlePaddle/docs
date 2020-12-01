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
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64、float16。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

        import paddle
        paddle.disable_static()

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.atan(x)
        print(out.numpy())
        # [-0.38050638 -0.19739556  0.09966865  0.29145679]
