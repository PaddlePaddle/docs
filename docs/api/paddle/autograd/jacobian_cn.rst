.. _cn_api_paddle_autograd_jacobian:

jacobian
-------------------------------

.. py:class:: paddle.autograd.jacobian(ys, xs, batch_axis=None)

计算因变量 ``ys`` 对 自变量 ``xs`` 的雅可比矩阵。

其中 ``ys`` 表示 ``xs`` 经过某种运算得到的输出， ``ys`` 和 ``xs`` 可以是 Tensor 或 Tensor 元组， ``batch_axis`` 表示参数数据的 batch 维度的位置。

当输入为 Tensor 元组时，返回结果为具有与 ``xs`` 相同嵌套层数的 ``Jacobian`` 对象，且每个 Jacobian 的形状与 ``xs`` 元组一一对应相同。

- 在 ``batch_axis=None`` 时，只支持 0 维 Tensor 或 1 维 Tensor，假设 ``xs`` 的形状为 ``[N, ]`` ， ``ys`` 的形状为 ``[M, ]``，则最终输出雅可比矩阵形状为 ``[M, N]`` 。

- 在 ``batch_axis=0`` 时，只支持 1 维 Tensor 或 2 维 Tensor，假设 ``xs`` 的形状为 ``[B, N]``， ``ys`` 的形状为 ``[B, M]``，则最终输出雅可比矩阵形状为 ``[B, M, N]`` 。

``Jacobian`` 对象被创建后，并没有发生实际的计算过程，而是采用惰性求值方法进行计算，可以对其进行多维索引来获取整个雅可比矩阵或子矩阵，此时会进行实际求值计算并返回结果。同时在实际求值的过程中，会对计算完毕的子矩阵进行缓存，以避免在后续的索引过程中产生重复计算。

例如，假设 ``Jacobian`` 的实例 ``J`` 形状为 ``[B, M, N]``，假设 ``M > 4`` ,
则 ``J[:, 1:4:1, :]`` 表示获取 ``J`` 的第 ``1`` 行到第 ``3`` 行值。实际计算时，仅会对
第 ``1`` 行到第 ``3`` 进行求值，并且 ``1`` 到 ``3`` 行的计算结果会以行的粒度进行缓存，下次再
获取上述某一行或多行结果时，已经计算过的部分不会被重复计算。

更多索引方式可以参考 Paddle 官网 `索引和切片 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#suoyinheqiepian>`_ 。

.. note::
  当前暂不支持省略号索引，且暂时只支持 ``batch_axis=None`` 和 ``batch_axis=0``。

参数
:::::::::

- **ys** (Tensor|Tuple[Tensor, ...]) - 因变量 ``ys`` ，数据类型为 Tensor 或 Tensor 元组。
- **xs** (Tensor|Tuple[Tensor, ...]) - 自变量 ``xs`` ，数据类型为 Tensor 或 Tensor 元组。
- **batch_axis** (int，可选) - ``0`` 表示参数包含 batch 维度，且第 0 维为 batch 维，
  ``None`` 表示参数不包含 batch。默认值为 ``None`` 。

返回
:::::::::

用于计算雅可比矩阵的 ``Jacobian`` 实例。

代码示例
:::::::::

COPY-FROM: paddle.autograd.jacobian
