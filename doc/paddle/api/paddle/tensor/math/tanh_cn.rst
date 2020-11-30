.. _cn_api_tensor_tanh:

tanh
-------------------------------

.. py:function:: paddle.tanh(x, name=None)


tanh 激活函数

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数:

    - **x** (Tensor) - Tanh算子的输入, 多维Tensor，数据类型为 float16，float32或float64。
    - **name** (str, 可选) - 该层名称（可选，默认为None）。具体用法请参见 :ref:`api_guide_Name`。

返回: tanh的输出Tensor，和输入有着相同类型和shape。

返回类型: Tensor

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
    y = paddle.tanh(x)
    print(y) # y=[-0.37994900, -0.19737528, 0.09966799, 0.29131261]