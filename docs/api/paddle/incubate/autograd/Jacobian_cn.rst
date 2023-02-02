.. _cn_api_paddle_incubate_autograd_Jacobian:

Jacobian
-------------------------------

.. py:class:: paddle.incubate.autograd.Jacobian(func, xs, is_batched=False)

计算函数 ``func`` 在 ``xs`` 处的雅可比矩阵。

其中，函数 ``func`` 的输入、输出可以为 Tensor 或 Tensor 序列，``is_batched=True`` 表示是否支
持 batch, ``True`` 表示输入和输出的第零维是 batch。

在计算雅可比矩阵时，输入 Tensor batch 维外的其它维度会被展平，且当输入为 Tensor 序列时，
所有展平后的 Tensor 会被拼接成一个新的 Tensor。输出按照同样规则进行处理。因此，``Jacobian`` 最终
的输出为一个二维(不包含 batch)或三维(包含 batch，第零维为 batch)的 Tensor。

例如，假设 ``is_batched=True``，输入 Tensor 经过展平并拼接后的形状为 ``(B, M)``，输出
Tensor 经过展平并拼接后的形状为 ``(B, N)``，则最终输出雅可比矩阵形状为 ``(B, M, N)`` 。
其中，``B`` 为 batch 维大小，``M`` 为展平并拼接后的输入大小，``N`` 为展平并拼接后的输出大小。

``Jacobian`` 对象被创建后，并没有发生实际的计算过程，而是采用惰性求值方法进行计算，可以通过
对 ``Jacobian`` 多维索引获取整个雅可比矩阵或子矩阵的实际结果，并且实际计算也发生在这一过程，已
经计算的子矩阵也会被缓存以避免重复计算。

例如，假设 ``Jacobian`` 的实例 ``J`` 形状为 ``(B, M, N)``，假设 ``M>4`` ,
则 ``J[:, 1:4:1, :]`` 表示获取 ``J`` 的第 ``1`` 行到第 ``3`` 行值，实际计算时，仅会对
第 ``1`` 行到第 ``3`` 进行求值，并且 ``1`` 到 ``3`` 行的计算结果会以行的粒度进行缓存，下次再
获取上述某一行或多行结果时不会发生重复计算。

更多索引方式可以参考 Paddle 官网 `索引和切片 <https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html#suoyinheqiepian>`_ 。

.. note::
  当前暂不支持省略号索引。

.. warning::
  该 API 目前为 Beta 版本，函数签名在未来版本可能发生变化。

参数
:::::::::

- **func** (Callable) - Python 函数，输入参数为 ``xs``，输出为 Tensor 或 Tensor 序列。
- **xs** (Tensor|Sequence[Tensor]） - 函数 ``func`` 的输入参数，数据类型为 Tensor 或
  Tensor 序列。
- **is_batched** (bool) - ``True`` 表示包含 batch 维，且默认第零维为 batch 维，``False``
  表示不包含 batch。默认值为 ``False`` 。

返回
:::::::::

用于计算雅可比矩阵的 ``Jacobian`` 实例。

代码示例
:::::::::

COPY-FROM: paddle.incubate.autograd.Jacobian
