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
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64 、float16。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor

**代码示例**：

.. code-block:: python

        import paddle
        paddle.disable_static()

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.ceil(x)
        print(out.numpy())
        # [-0. -0.  1.  1.]
