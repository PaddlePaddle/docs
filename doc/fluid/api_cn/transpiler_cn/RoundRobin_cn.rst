.. _cn_api_fluid_transpiler_RoundRobin:

RoundRobin
-------------------------------

:api_attr: 声明式编程(静态图)专用API

.. py:class:: paddle.fluid.transpiler.RoundRobin(pserver_endpoints)

该方法使用 ``RoundRobin`` 的方式将变量散列到多个parameter server终端。

`RondRobin <https://en.wikipedia.org/wiki/Round-robin_scheduling>`_

参数:
  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 

返回：实例化后的RoundRobin的对象

返回类型：RoundRobin

**代码示例**

.. code-block:: python

          import paddle.fluid.transpiler.RoundRobin as RoundRobin

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)


.. py:method:: dispatch(varlist)

该方法使用RoundRobin的方式将多个参数散列到多个parameter Server终端。

参数:
  - **varlist** (list) - 参数 (var1, var2, var3) 的 list

返回：基于varlist中var的顺序，返回参数服务器(ip:port)的列表， 列表中的数据量和varlist的数据量一致。

返回类型：list

**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.dispatch(vars)


.. py:method:: reset()

该方法将重置RoundRobin内置的计数， 计数将重置为0。

返回：无。

**代码示例**

.. code-block:: python

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = RoundRobin(pserver_endpoints)
          rr.reset()


