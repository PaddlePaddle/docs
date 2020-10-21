.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)

:alias_main: paddle.cos
:alias: paddle.cos,paddle.tensor.cos,paddle.tensor.math.cos
:old_api: paddle.fluid.layers.cos



余弦函数。

输入范围是 `(-inf, inf)` ， 输出范围是 `[-1,1]`。

.. math::

    out = cos(x)

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64 、float16。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型：Tensor

**代码示例**：

.. code-block:: python

        import paddle
        paddle.disable_static()

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cos(x)
        print(out.numpy())
        # [0.92106099 0.98006658 0.99500417 0.95533649]
