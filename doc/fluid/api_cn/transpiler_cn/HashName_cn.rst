.. _cn_api_fluid_transpiler_HashName:

HashName
-------------------------------

.. py:class:: paddle.fluid.transpiler.HashName(pserver_endpoints)

使用 python ``Hash()`` 函数将变量名散列到多个pserver终端。

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 

**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)




