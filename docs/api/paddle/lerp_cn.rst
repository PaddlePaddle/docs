.. _cn_api_paddle_tensor_lerp:

lerp
-------------------------------

.. py:function:: paddle.lerp(x, y, weight, name=None)
基于给定的 weight 计算 x 与 y 的线性插值

.. math::
    lerp(x, y, weight) = x + weight * (y - x)
参数
:::::::::

- **x**  (Tensor) - 输入的 Tensor，作为线性插值开始的点，数据类型为：bfloat16、float16、float32、float64。
- **y**  (Tensor) - 输入的 Tensor，作为线性插值结束的点，数据类型为：bfloat16、float16、float32、float64。
- **weight**  (float|Tensor) - 给定的权重值，weight 为 Tensor 时数据类型为：bfloat16、float16、float32、float64。
- **name**  (str，可选） - 操作的名称(可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::

输出 Tensor，与 ``x`` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.lerp
