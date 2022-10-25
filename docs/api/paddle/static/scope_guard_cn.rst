.. _cn_api_fluid_executor_scope_guard:

scope_guard
-------------------------------


.. py:function:: paddle.static.scope_guard (scope)





通过 python 的 ``with`` 语句切换作用域（scope）。
作用域记录了变量名和变量 ( :ref:`api_guide_Variable` ) 之间的映射关系，类似于编程语言中的大括号。
如果未调用此接口，所有的变量和变量名都会被记录在默认的全局作用域中。
当用户需要创建同名的变量时，如果不希望同名的变量映射关系被覆盖，则需要通过该接口切换作用域。
通过 ``with`` 语句切换后，``with`` 语句块中所有创建的变量都将分配给新的作用域。

参数
::::::::::::

  - **scope** (Scope) - 新的作用域。

返回
::::::::::::

无。

代码示例
::::::::::::

COPY-FROM: paddle.static.scope_guard
