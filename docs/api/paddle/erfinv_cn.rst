.. _cn_api_paddle_tensor_erfinv:

erfinv
-------------------------------

.. py:function:: paddle.erfinv(x, name=None)
计算输入矩阵 x 的逆误差函数。
请参考 erf 计算公式 :ref:`cn_api_fluid_layers_erf`

.. math::
    erfinv(erf(x)) = x

参数
:::::::::

- **x**  (Tensor) - 输入的 Tensor，数据类型为：float16、bfloat16、float32、float64。
- **name**  (str，可选) - 操作的名称（可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出 Tensor，与 :attr:`x` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.erfinv
