.. _cn_api_paddle_nn_Identity:

Identity
-------------------------------

.. py:class:: paddle.nn.Identity(*args, **kwargs)


**等效层**。对于输入 Tensor :math:`X`，计算公式为：

.. math::

    Out = X


参数
:::::::::

- **args** - 任意的参数（没有使用）
- **kwargs** – 任意的关键字参数（没有使用）

形状
:::::::::

- 输入：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维 Tensor。
- 输出：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维 Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.Identity
