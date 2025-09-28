.. _cn_api_paddle_incubate_autograd_jvp:

jvp
-------------------------------

.. py:function:: paddle.incubate.autograd.jvp(func, xs, v=None)

计算函数 ``func`` 在 ``xs`` 处的雅可比矩阵与向量 ``v`` 的乘积。

.. warning::
  该 API 目前为 Beta 版本，函数签名在未来版本可能发生变化。

参数
:::::::::

- **func** (Callable) - Python 函数，输入参数为 ``xs``，输出为 Tensor 或 Tensor 序列。
- **xs** (Tensor|Sequence[Tensor]） - 函数 ``func`` 的输入参数，数据类型为 Tensor 或
  Tensor 序列。
- **v** (Tensor|Sequence[Tensor]|None，可选) - 用于计算 ``jvp`` 的输入向量，形状要求
  与 ``xs`` 一致。默认值为 ``None``，即相当于形状与 ``xs`` 一致，值全为 1 的 Tensor 或
  Tensor 序列。

返回
:::::::::

- **func_out** (Tensor|tuple[Tensor]) - 函数 ``func(xs)`` 的输出。
- **jvp** (Tensor|tuple[Tensor]) - ``jvp`` 计算结果。

代码示例
:::::::::

COPY-FROM: paddle.incubate.autograd.jvp
