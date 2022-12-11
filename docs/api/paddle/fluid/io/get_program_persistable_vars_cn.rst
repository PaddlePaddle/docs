.. _cn_api_fluid_io_get_program_persistable_vars:

get_program_persistable_vars
-------------------------------

.. py:function:: paddle.fluid.io.get_program_persistable_vars(program)




该接口从Program中获取所有persistable的变量。

参数
::::::::::::

 - **program**  ( :ref:`cn_api_fluid_Program` ) – 从该Program中获取persistable的变量。

返回
::::::::::::
 包含此Program中所有persistable的变量

返回类型
::::::::::::
 list

代码示例
::::::::::::

COPY-FROM: paddle.fluid.io.get_program_persistable_vars