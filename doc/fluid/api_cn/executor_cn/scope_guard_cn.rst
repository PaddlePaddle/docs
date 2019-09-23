.. _cn_api_fluid_executor_scope_guard:

scope_guard
-------------------------------

.. py:function:: paddle.fluid.executor.scope_guard (scope)


切换全局/默认作用域（scope）。模型运行时，所有变量都将被放到新的scope中。

参数：
    - **scope** - 新的全局/默认作用域（scope）。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    new_scope = fluid.Scope()
    with fluid.scope_guard(new_scope):
         fluid.global_scope().var("data").get_tensor().set(numpy.ones((1, 2)), fluid.CPUPlace())
    data = numpy.array(new_scope.find_var("data").get_tensor())
    print(data)  # [[1. 1.]]]

