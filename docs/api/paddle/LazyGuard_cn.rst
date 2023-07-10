.. _cn_api_fluid_LazyGuard:

LazyGuard
-------------------------------

.. py:class:: paddle.LazyGuard()



LazyGuard 是一个用于设置模型（继承自 ``paddle.nn.Layer`` ） 中参数延迟初始化的上下文管理器。配合使用 python 的 ``with`` 语句来将 ``with LazyGuard():`` 代码块下所有模型在实例化时，其内部的参数均不会立即申请内存空间。


代码示例
::::::::::::

COPY-FROM: paddle.LazyGuard