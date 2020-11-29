.. _cn_api_fluid_layers_atan:

atan
-------------------------------

.. py:function:: paddle.atan(x, name=None)




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

        x = paddle.to_tensor([-0.8183,  0.4912, -0.6444,  0.0371])
        res = paddle.atan(x)
        print(res)
        # [-0.6858,  0.4566, -0.5724,  0.0371]
