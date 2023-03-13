.. _cn_api_fluid_io_get_program_persistable_vars:

get_program_persistable_vars
-------------------------------

.. py:function:: paddle.fluid.io.get_program_persistable_vars(program)




该接口从 Program 中获取所有 persistable 的变量。

参数
::::::::::::

 - **program**  ( :ref:`cn_api_fluid_Program` ) – 从该 Program 中获取 persistable 的变量。

返回
::::::::::::
 包含此 Program 中所有 persistable 的变量

返回类型
::::::::::::
 list

代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.get_program_persistable_vars
