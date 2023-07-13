.. _cn_api_paddle_autograd_hessian:

hessian
-------------------------------

.. py:class:: paddle.autograd.hessian(ys, xs, batch_axis=None)

计算因变量 ``ys`` 对 自变量 ``xs`` 的海森矩阵。

其中 ``ys`` 表示 ``xs`` 经过某种运算得到的输出， ``ys`` 只能是单个 Tensor， ``xs`` 可以是 Tensor 或 Tensor 元组，``batch_axis`` 表示参数数据的 batch 维度的位置。

当输入 ``xs`` 为 Tensor 元组时，返回结果为 ``Hessian`` 元组，假设 ``xs`` 元组的内部形状构成为 ``([M1, ], [M2, ])``，则返回结果的形状构成 ``(([M1, M1], [M1, M2]), ([M2, M1], [M2, M2]))``

- 在 ``batch_axis=None`` 时，只支持 0 维 Tensor 或 1 维 Tensor，假设 ``xs`` 的形状为 ``[N, ]``， ``ys`` 的形状为 ``[]`` (0 维 Tensor)，则最终输出单个海森矩阵，其形状为 ``[N, N]`` 。

- 在 ``batch_axis=0`` 时，只支持 1 维 Tensor 或 2 维 Tensor，假设 ``xs`` 的形状为 ``[B, N]``， ``ys`` 的形状为 ``[B, ]``，则最终输出雅可比矩阵形状为 ``[B, N, N]`` 。

``Hessian`` 对象被创建后，并没有发生完整的计算过程，而是采用部分惰性求值方法进行计算，可以对其进行多维索引来获取整个海森矩阵或子矩阵，此时会进行实际求值计算并返回结果。同时在实际求值的过程中，会对计算完毕的子矩阵进行缓存，以避免在后续的索引过程中产生重复计算。

更多索引方式可以参考 Paddle 官网 `索引和切片 <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#suoyinheqiepian>`_ 。

.. note::
  当前暂不支持省略号索引，且暂时只支持 ``batch_axis=None`` 和 ``batch_axis=0``。

参数
:::::::::

- **ys** (Tensor) - 因变量 ``ys`` ，数据类型为单个 Tensor。
- **xs** (Tensor|Tuple[Tensor, ...]) - 自变量 ``xs`` ，数据类型为 Tensor 或 Tensor 元组。
- **batch_axis** (int，可选) - ``0`` 表示参数包含 batch 维度，且第 0 维为 batch 维，
  ``None`` 表示参数不包含 batch。默认值为 ``None`` 。

返回
:::::::::

用于计算海森矩阵的 ``Hessian`` 实例。

代码示例
:::::::::

COPY-FROM: paddle.autograd.hessian
