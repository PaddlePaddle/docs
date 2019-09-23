.. _cn_api_fluid_scope_guard:

scope_guard
-------------------------------

.. py:function:: paddle.fluid.scope_guard(scope)


该接口通过 python 的 ``with`` 语句修改全局或默认的作用域（scope），修改后，运行时中的所有变量都将分配给新的作用域。

参数：
  - **scope** (Scope) - 新的全局或默认的作用域。

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid
  import numpy
  
  new_scope = fluid.Scope()
  with fluid.scope_guard(new_scope):
       fluid.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), fluid.CPUPlace())
  numpy.array(new_scope.find_var("data").get_tensor())  # array([[1., 1.], [1., 1.]])
 




