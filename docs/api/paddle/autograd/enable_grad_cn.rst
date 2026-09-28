.. _cn_api_paddle_autograd_enable_grad:

enable_grad
-------------------------------

.. py:class:: paddle.autograd.enable_grad()


创建一个上下文来启用动态图梯度计算。在此模式下，每次计算的结果都将具有 stop_gradient=False。

也可以用作一个装饰器（需要创建实例对象作为装饰器）。

该 API 与 :ref:`cn_api_paddle_enable_grad` 功能一致。

代码示例
::::::::::::

COPY-FROM: paddle.autograd.enable_grad
