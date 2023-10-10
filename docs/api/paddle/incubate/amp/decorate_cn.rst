.. _cn_api_paddle_incubate_amp_decorate:

decorate
-------------------------------

.. py:function:: paddle.incubate.amp.decorate(optimizer)


将给定的优化器包装为 `OptimizerWithSparsityGuarantee`。如果在动态图模式下运行，装饰时 ASP 会为支持的参数创建掩码变量。如果在静态图模式下运行，ASP 会在调用 minimize() 时创建掩码变量并插入必要的掩码操作。


参数
:::::::::

**optimizer** (Optimizer) – 用于模型训练的优化器。

返回
:::::::::

**OptimizerWithSparsityGuarantee** - 一个用于 ASP 的包装器，用于装饰给定优化器的 minimize() 或者 step()。

代码示例
:::::::::

1. 动态图模式

COPY-FROM: paddle.incubate.amp.decorate:dynamic_graph

2. 静态图模式

COPY-FROM: paddle.incubate.amp.decorate:static_graph
