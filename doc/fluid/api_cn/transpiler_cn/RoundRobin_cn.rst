.. _cn_api_fluid_transpiler_RoundRobin:

RoundRobin
-------------------------------

.. py:class:: paddle.fluid.transpiler.RoundRobin(pserver_endpoints)

使用 ``RondRobin`` 方法将变量分配给服务器端点。

`RondRobin <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 
 
**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)




