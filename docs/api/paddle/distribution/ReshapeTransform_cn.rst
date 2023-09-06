.. _cn_api_paddle_distribution_ReshapeTransform:

ReshapeTransform
-------------------------------

.. py:class:: paddle.distribution.ReshapeTransform(in_event_shape, out_event_shape)

``ReshapeTransform`` 将输入 Tensor 的事件形状 ``in_event_shape`` 改变为 ``out_event_shape``。其中，``in_event_shape``、``out_event_shape`` 需要包含相同的元素个数。


参数
:::::::::

- **in_event_shape** (Sequence[int]) - Reshape 前的事件形状。
- **out_event_shape** (float|Tensor） - Reshape 后的事件形状。


代码示例
:::::::::

COPY-FROM: paddle.distribution.ReshapeTransform

方法
:::::::::

forward(x)
'''''''''

计算正变换 :math:`y=f(x)` 的结果。

**参数**

- **x** (Tensor) - 正变换输入参数，通常为 :ref:`cn_api_distribution_Distribution` 的随机采样结果。

**返回**

Tensor，正变换的计算结果。


inverse(y)
'''''''''

计算逆变换 :math:`x = f^{-1}(y)`。

**参数**

- **y** (Tensor) - 逆变换的输入参数。

**返回**

Tensor，逆变换的计算结果。

forward_log_det_jacobian(x)
'''''''''

计算正变换雅可比行列式绝对值的对数。

如果变换不是一一映射，则雅可比矩阵不存在，抛出 ``NotImplementedError``。

**参数**

- **x** (Tensor) - 输入参数。

**返回**

Tensor，正变换雅可比行列式绝对值的对数。


inverse_log_det_jacobian(y)
'''''''''

计算逆变换雅可比行列式绝对值的对数。

与 ``forward_log_det_jacobian`` 互为负数。

**参数**

- **y** (Tensor) - 输入参数。

**返回**

Tensor，逆变换雅可比行列式绝对值的对数。


forward_shape(shape)
'''''''''

推断正变换输出形状。

**参数**

- **shape** (Sequence[int]) - 正变换输入的形状。

**返回**

Sequence[int]，正变换输出的形状。


inverse_shape(shape)
'''''''''

推断逆变换输出形状。

**参数**

- **shape** (Sequence[int]) - 逆变换输入的形状。

**返回**

Sequence[int]，逆变换输出的形状。
