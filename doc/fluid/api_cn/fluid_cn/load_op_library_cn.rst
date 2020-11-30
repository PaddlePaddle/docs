.. _cn_api_fluid_load_op_library:

load_op_library
-------------------------------

.. py:class:: paddle.fluid.load_op_library

:api_attr: 声明式编程模式（静态图)



``load_op_library`` 用于自定义C++算子中，用来加载算子动态共享库。加载库后，注册好的算子及其Kernel实现将在PaddlePaddle主进程中可以被调用。 请注意，自定义算子的类型不能与框架中的现有算子类型相同。

参数：
    - **lib_filename** (str) – 动态共享库的名字。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       #fluid.load_op_library('custom_op.so')




