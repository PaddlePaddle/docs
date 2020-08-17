.. _cn_api_fluid_ComplexVariable:

ComplexVariable
-------------------------------


.. py:class:: paddle.fluid.ComplexVariable(real, imag)

:api_attr: 命令式编程模式（动态图)

``ComplexVariable`` 可以定义存储复数的变量。它包含两个参数 ``real`` 和 ``imag`` ,分别存储复数的实数部分与虚数部分。

.. note::
``ComplexVariable`` 不应该被直接调用。目前只支持动态图模式，请通过给  :ref:`cn_api_fluid_dygraph_to_variable` 传入复数数据的方式创建一个动态图下的复数变量。

参数:
    - **real** (Variable) - 存储复数的实数部分。
    - **imag** (Variable) - 存储复数的虚数部分

**代码示例**

.. code-block:: python
   
    import paddle.fluid as fluid
    import numpy as np

    a = np.array([1.0+2.0j, 0.2])
    with fluid.dygraph.guard():
        var = fluid.dygraph.to_variable(a, name="new_var")
        print(var.name, var.dtype, var.shape)
        # ({'real': u'new_var.real', 'imag': u'new_var.imag'}, 'complex128', [2L])
        print(var.numpy())
        # [1. +2.j 0.2+0.j]
