.. _cn_api_paddle_no_grad:

no_grad
-------------------------------

.. py:class:: paddle.no_grad()



创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有 stop_gradient=True。

也可以用作一个装饰器（需要创建实例对象作为装饰器）。

代码示例
::::::::::::

COPY-FROM: paddle.no_grad
