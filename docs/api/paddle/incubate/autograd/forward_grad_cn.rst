.. _cn_api_paddle_incubate_autograd_forward_grad:

forward_grad
-------------------------------

.. py:function:: paddle.incubate.autograd.forward_grad(outputs, inputs, grad_inputs=None)

前向模式自动微分。

.. warning::
  该 API 目前为 Beta 版本，函数签名在未来版本可能发生变化。

.. note::
  仅支持静态图模式


参数
:::::::::

- **outputs** (Tensor|Sequence[Tensor]） - 输出 Tensor 或 Tensor 序列。
- **inputs** (Tensor|Sequence[Tensor]) - 输入 Tensor 或 Tensor 序列。
- **grad_inputs** (Tensor|Sequence[Tensor], 可选) - 输入的初始梯度，形状与输入相同。默认值为 None,表示形状与输入相同，值全为 1 的 Tensor 或 Tensor 列表

返回
:::::::::

- **grad_outputs** (Tensor|tuple[Tensor]) - 输出梯度。

代码示例
:::::::::

COPY-FROM: paddle.incubate.autograd.forward_grad
