.. _cn_api_fluid_layers_cosh:

cosh
-------------------------------

.. py:function:: paddle.cosh(x, name=None)




双曲余弦函数。

输入范围是 `(-inf, inf)` ， 输出范围是 `[1,inf]`。

.. math::

    out = \frac{exp(x)+exp(-x)}{2}

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64 、float16。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

**代码示例**：

.. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cosh(x)
        print(out)
        # [1.08107237 1.02006674 1.00500417 1.04533851]
