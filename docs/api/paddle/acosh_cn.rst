.. _cn_api_fluid_layers_acosh:

acosh
-------------------------------

.. py:function:: paddle.acosh(x, name=None)




Arccosh函数。

.. math::
    out = acosh(x)

参数:
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：输出Tensor，与 ``x`` 维度相同、数据类型相同。

返回类型： Tensor


**代码示例**：

.. code-block:: python

        import paddle

        x = paddle.to_tensor([1., 3., 4., 5.])
        out = paddle.acosh(x)
        print(out)
        # [0.        , 1.76274729, 2.06343699, 2.29243159]
