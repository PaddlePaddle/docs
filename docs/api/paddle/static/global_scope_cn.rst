.. _cn_api_paddle_static_global_scope:

global_scope
-------------------------------


.. py:function:: paddle.static.global_scope()




获取全局/默认作用域实例。很多 API 使用默认 ``global_scope``，例如 ``Executor.run`` 等。

返回
::::::::::::
Scope，全局/默认作用域实例。

代码示例
::::::::::::

COPY-FROM: paddle.static.global_scope
