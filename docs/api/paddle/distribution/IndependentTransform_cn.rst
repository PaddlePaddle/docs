.. _cn_api_paddle_distribution_IndependentTransform:

IndependentTransform
-------------------------------

.. py:class:: paddle.distribution.IndependentTransform(base, reinterpreted_batch_rank)


 ``IndependentTransform`` 将一个基础变换 :attr:`base` 的部分批（batch）维度 ``reinterpreted_batch_rank`` 扩展为事件（event）维度。

 ``IndependentTransform`` 不改变基础变换 ``forward`` 以及 ``inverse`` 计算结果，但会对 ``forward_log_det_jacobian`` 以及 ``inverse_log_det_jacobian`` 计算结果沿着扩展的维度进行求和。

例如，假设基础变换为 ``ExpTransform``，其输入为一个随机采样结果 ``x``，形状为 :math:`(S=[4], B=[2,2], E=[3])` , :math:`S`、:math:`B`、:math:`E` 分别表示采样形状、批形状、事件形状，``reinterpreted_batch_rank=1``。则 ``IndependentTransform(ExpTransform)`` 变换后，``x`` 的形状为 :math:`(S=[4], B=[2], E=[2,3])`，即将最右侧的批维度作为事件维度。此时 ``forward`` 和 ``inverse`` 输出形状仍是 :math:`[4, 2, 2, 3]`，但 ``forward_log_det_jacobian`` 以及 ``inverse_log_det_jacobian`` 输出形状为 :math:`[4, 2]`。


参数
:::::::::

- **base** (Transform) - 基础变换。
- **reinterpreted_batch_rank** (int) - 被扩展为事件维度的最右侧批维度数量，需大于 0。


代码示例
:::::::::

COPY-FROM: paddle.distribution.IndependentTransform

方法
:::::::::

forward(x)
'''''''''

计算正变换 :math:`y=f(x)` 的结果。

**参数**

- **x** (Tensor) - 正变换输入参数，通常为 :ref:`cn_api_paddle_distribution_Distribution` 的随机采样结果。

**返回**

Tensor，正变换的计算结果。


inverse(y)
'''''''''

计算逆变换 :math:`x = f^{-1}(y)`

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
