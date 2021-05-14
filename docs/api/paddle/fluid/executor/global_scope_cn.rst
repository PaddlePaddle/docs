.. _cn_api_fluid_executor_global_scope:

global_scope
-------------------------------


.. py:function:: paddle.static.global_scope()




获取全局/默认作用域实例。很多API使用默认 ``global_scope`` ，例如 ``Executor.run`` 等。

返回：全局/默认作用域实例

返回类型：Scope

**示例代码**

.. code-block:: python

        import paddle
        import numpy

        paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((1, 2)), paddle.CPUPlace())
        data = numpy.array(paddle.static.global_scope().find_var("data").get_tensor())
        print(data)  # [[1. 1.]]
