.. _cn_api_fluid_transpiler_HashName:

HashName
-------------------------------


.. py:class:: paddle.fluid.transpiler.HashName(pserver_endpoints)




该方法使用 python ``Hash()`` 函数将变量散列到多个parameter server节点。

参数
::::::::::::

  - **pserver_endpoints** (list) - endpoint （ip:port）的 list 

返回
::::::::::::
实例化后的HashName的对象

返回类型
::::::::::::
HashName

代码示例
::::::::::::

.. code-block:: python

          import paddle.fluid.transpiler.HashName as HashName

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = HashName(pserver_endpoints)
          rr.dispatch(vars)


方法
::::::::::::
reset()
'''''''''

该方法将重置HashName内置的计数，计数将重置为0。

**返回**
无。

**代码示例**

.. code-block:: python

          import paddle.fluid.transpiler.HashName as HashName 

          pserver_endpoints = [“127.0.0.1:6007”, “127.0.0.1:6008”]
          vars = [“var1”,”var2”,”var3”,”var4”,”var5”]
          rr = HashName(pserver_endpoints)
          rr.reset()

