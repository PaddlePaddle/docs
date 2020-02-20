.. _cn_api_fluid_load_op_library:

load_op_library
-------------------------------

.. py:class:: paddle.fluid.load_op_library

``load_op_library`` 用于加载动态库，包括自定义运算符和内核。 加载库后，注册好的操作和内核将在PaddlePaddle主进程中可以被调用。 请注意，自定义运算符的类型不能与框架中的现有运算符相同。

参数：
    - **lib_filename** (str) – 动态库的名字。

**代码示例**

.. code-block:: python

       import paddle.fluid as fluid
       #fluid.load_op_library('custom_op.so')



