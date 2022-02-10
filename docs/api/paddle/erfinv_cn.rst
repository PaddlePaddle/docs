.. _cn_api_paddle_tensor_erfinv:

erfinv
-------------------------------

.. py:function:: paddle.erfinv(x)
计算输入矩阵x的逆误差函数。
请参考erf计算公式 :ref:`cn_api_fluid_layers_erf`

.. math::
    erfinv(erf(x)) = x

参数
:::::::::

- **x**  (Tensor) - 输入的Tensor，数据类型为：float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出Tensor，与 ``x`` 数据类型相同。

代码示例
:::::::::

.. code-block:: python

        import paddle
        
        x = paddle.to_tensor([0, 0.5, -1.], dtype="float32")
        out = paddle.erfinv(x)
        # out: [0, 0.4769, -inf]
