.. _cn_api_paddle_distribution_SoftmaxTransform:

SoftmaxTransform
-------------------------------

.. py:class:: paddle.distribution.SoftmaxTransform

Softmax 变换，首先进行 :math:`y = exp(x)` 变换，然后归一化。

Softmax 变换将向量变换为单纯形。

.. note::
   Softmax 不是双射函数，所以 ``forward_log_det_jacobian``、``inverse_log_det_jacobian`` 未实现。

代码示例
:::::::::

COPY-FROM: paddle.distribution.SoftmaxTransform

方法
:::::::::

forward(x)
'''''''''

计算正变换 :math:`y=f(x)` 的结果。

**参数**

- **x** (Tensor) - 正变换输入参数，通常为 :ref:`cn_api_distribution_Distribution` 的随机采样结果。

**返回**

- **y** (Tensor) - 正变换的计算结果。


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

如果变换不是一一映射，则雅可比矩阵不存在，返回 ``NotImplementedError`` 。

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
